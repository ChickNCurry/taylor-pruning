# emb-ml-challenge

### tasks done

- perform inference on gpu
- model.eval(), torch.no_grad() (torch.inference_mode() doesn’t work)
- added yolo_pruned class for loading pruned weights via state_dict hook
- modified structured pruning functions for yolo (adaption to bn)
- added yolo_fused class for fused conv and bn layers
- taylor-based pruning
- optimize rest of camera loop
- compile and run with onnx
- run iterative pruning to get different model sizes
- what didnt work
    - modified yolo for single class (person) and finetuned on person only dataset
    - nms + filter boxes due to torhciviosn version incompatibility
    - pruned all layers with ratio=0.3 (except last layer) and finetuned because taylor better

### tasks todo

- fix torchvision / pytoch compatibility for filter boxes + nms
- evaluate with precision, recall, f1score, intersection over union
- migrate to py instead of notebooks and run without jupyter (web server)
- prune / finetune with all classes
- pruning based on taylor criterion of fusedconvbn, but prune seperate layers
- remove other classes before filter boxes


### optional todos

- try out different pruning ratios
- try out pruning on different layers
- feature map-based pruning

### presentation ideas

- show loop profile to highlight bottlenecks
- show what we changed
- show 3 variants
    - lower fps but higher precision / recall
    - medium fps …
    - higher fps …
- graphs of metrics for each
- live demo
- finetuned for single classe person for 20 epochs
- iterative taylor pruning, per iteration (10), do
    - N = num of channels to pruned based on model size * multipler (10) (large pruning steps)
    - forwardpass to gather activations and gradients
    - compute taylor criterion to find N least relevant channels
    - structured prune N channels (can be all throughout model but mostly in later layers)
    - finetune last layer for 10 epochs with low learing rate (0.0001) to recover ap (long finetuning)
- found important: large pruning steps, long finetuning with low learning rate

### presentation flow

1. finetuned to single class (person) for 20 epochs -> ap of 0.66
2. profiling -> focus on pruning
3. taylor pruning explanation
    - forward pass to gather feature maps and gradients
    - channel saliencies based on taylor criterion
    - remove specific number of channels with lowest saliency
    - no need to specify which layers to prune, taylor pruning tends to select last layers
        - show how many channels were pruned per layer -> focus on last layers
    - compare taylor vs l1 pruning
4. iterative pruning for 10 iteration
    - taylor pruning, removing amount of channels proportional to model size
    - with finetuning for 10 epochs in each iteration to regain ap 
        - much lower learning rate
    - show pruning run graphs
6. additional speedups: 
    - fuse conv and bn, filter boxes + nms, onnx compiling, quantization?
    - deployed in docker container with webserver
7. chose 3 models
        1. ... ap and ... model size -> fps of ...
        2. ... ap and ... model size -> fps of ...
        3. ... ap and ... model size -> fps of ...


# logging

```
sudo systemctl stop nvargus-daemon
export enableCamPclLogs=5
export enableCamScfLogs=5
sudo nvargus-daemon
```
