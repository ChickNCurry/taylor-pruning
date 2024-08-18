import copy
from typing import Dict, List
import torch
import tqdm
import numpy as np
from heapq import nsmallest
from operator import itemgetter
import torch.nn.functional as F
import torch.nn as nn
from utils.tinyyolov2 import TinyYoloV2

# channels refer to output channels, which are feature maps of each layer
# pruning feature maps is pruning a stack of filters, which is a 3d kernel

class TinyYoloV2WithTaylorRanking(TinyYoloV2):
    def __init__(self, device, num_classes=1):
        super(TinyYoloV2WithTaylorRanking, self).__init__(num_classes=num_classes)

        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)
        self.register_buffer("anchors", torch.tensor(anchors))

        self.anchors_len = len(anchors)
        self.num_classes = num_classes
        self.device = device
        
        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self._register_load_state_dict_pre_hook(self._sd_hook)
        self.reset()
    
    def reset(self):
        self.channel_saliencies_per_layer = {}

    def forward(self, x, yolo=True, record_activations=False):
        self.activations = [] # size (8, B, C, H, W)
        self.gradients = [] # size (8, B, C, H, W)
        self.grad_index = 0 # up to 7

        x = self.conv1(x)
        if(record_activations): self._record_activation(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        if(record_activations): self._record_activation(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        if(record_activations): self._record_activation(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        if(record_activations): self._record_activation(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        if(record_activations): self._record_activation(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        if(record_activations): self._record_activation(x)
        x = self.bn6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        if(record_activations): self._record_activation(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        if(record_activations): self._record_activation(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                    torch.arange(nH, dtype=x.dtype, device=x.device),
                    torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW, #x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH, #y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None])/ nW, # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) /nH, # Height
                x[:, :, :, :, 4:5].sigmoid(), #confidence
                x[:, :, :, :, 5:].softmax(-1),], -1)
        
        return x
    
    def _sd_hook(self, state_dict, prefix, *_):
        for key in state_dict:
            if not ('conv' in key and 'weight' in key):
                continue
            
            n = int(key.split(".")[0][-1])

            dim_in = state_dict[f'conv{n}.weight'].shape[1]
            dim_out = state_dict[f'conv{n}.weight'].shape[0]

            conv = nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False)
            bn = nn.BatchNorm2d(dim_out)
            
            if n == 9:
                self.conv9 = nn.Conv2d(dim_in, self.anchors_len * (5 + self.num_classes), 1, 1, 0)
            else: 
                setattr(self, f"conv{n}", conv)
                setattr(self, f"bn{n}", bn)
        pass

    def _record_activation(self, x):
        x.register_hook(self._compute_channel_saliencies)
        self.activations.append(x)

    def _compute_channel_saliencies(self, grad):
        # -1 since indices start at 0
        layer_index = len(self.activations) - self.grad_index - 1

        # batches of multiple feature maps of shape (B, C, H, W)
        activation = self.activations[layer_index]

        # grad of shape (B, C, H, W) -> taylor of shape (B, C, H, W)
        taylor = activation * grad

        # per channel, average over all dims of batched feature maps -> channel ranks of shape (C)
        channel_saliencies = torch.abs(taylor.mean(dim=(0, 2, 3)))

        if layer_index not in self.channel_saliencies_per_layer:
            self.channel_saliencies_per_layer[layer_index] = torch.zeros(channel_saliencies.shape)

        self.channel_saliencies_per_layer[layer_index] += channel_saliencies
        self.grad_index += 1

    def _normalize_saliencies(self):
        for i in self.channel_saliencies_per_layer:

            saliencies = self.channel_saliencies_per_layer[i]

            # simple layer-wise l2 normalization -> sum over channels
            normalized_saliencies = saliencies / torch.sqrt(torch.sum(saliencies * saliencies))

            self.channel_saliencies_per_layer[i] = normalized_saliencies

    def get_lowest_ranking_channels(self, num_channels_to_prune):
        self._normalize_saliencies()

        tuples = []

        for l in sorted(self.channel_saliencies_per_layer.keys()):
            for c in range(self.channel_saliencies_per_layer[l].size(0)):

                saliency = self.channel_saliencies_per_layer[l][c].item()
                tuples.append((l, c, saliency))

        return nsmallest(num_channels_to_prune, tuples, itemgetter(2))


def generate_channel_saliencies(model, criterion, train_loader, device):
    model.reset()
    model.to(device)
    model.train()

    for _, (input, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)

        output = model(input, yolo=False, record_activations=True)
        loss, _ = criterion(output, target)

        loss.backward()
    

def tuples_to_channels_to_prune_per_layer(tuples: List[tuple]) -> Dict[int, List[int]]:
    channels_to_prune_per_layer = {}

    for i in range(1, 10):
        channels_to_prune_per_layer[i] = []

    # n+1 since tuples used layer index, dict will use layer number
    # we discard channel rankings
    for l, c, _ in tuples:
        channels_to_prune_per_layer[l+1].append(c)

    return channels_to_prune_per_layer


def get_channels_to_prune_per_layer(model, criterion, train_loader, device, num_channels_to_prune):
    generate_channel_saliencies(model, criterion, train_loader, device)

    channels_to_prune_tuples = model.get_lowest_ranking_channels(num_channels_to_prune)
    channels_to_prune_per_layer = tuples_to_channels_to_prune_per_layer(channels_to_prune_tuples)
    print([len(channels_to_prune_per_layer[k]) for k in channels_to_prune_per_layer.keys()])

    return channels_to_prune_per_layer


# feature maps are equal to output channels
def prune_state_dict(state_dict: Dict, channels_to_prune_per_layer: Dict[int, List[int]]) -> Dict:
    state_dict = copy.deepcopy(state_dict)
    
    for key, value in state_dict.items():
        n = key.split(".")[0][-1]  
        n = int(n) if n.isdigit() else None
        if n is None: continue
        
        if "conv" in key and "weight" in key:

            if n-1 in channels_to_prune_per_layer:
                channels = [i for i in range(value.shape[1]) if i not in channels_to_prune_per_layer[n-1]]
                value = value[:, channels, :, :]
            
            channels = [i for i in range(value.shape[0]) if i not in channels_to_prune_per_layer[n]]
            value = value[channels, :, :, :]
            state_dict[key] = value
        
        elif "bn" in key and "num_batches_tracked" not in key:
            channels = [i for i in range(value.shape[0]) if i not in channels_to_prune_per_layer[n]]
            value = value[channels]
            state_dict[key] = value
    
    return state_dict