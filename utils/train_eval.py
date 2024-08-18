import torch
import tqdm
from utils.ap import ap, display_roc, precision_recall_levels
from utils.yolo import filter_boxes, nms


def train(model, optimizer, criterion, device, train_loader, val_loader, val_before=True, num_epochs=15):
    val_aps = []
    avg_losses = []

    if val_before:
        val_ap = validate(model, device, val_loader, roc=False)
        val_aps.append(val_ap)

    for epoch in range(num_epochs):
        
        avg_loss = train_epoch(model, optimizer, criterion, device, train_loader)
        avg_losses.append(avg_loss)

        val_ap = validate(model, device, val_loader, roc=False)
        val_aps.append(val_ap)

    return val_aps, avg_losses


def train_epoch(model, optimizer, criterion, device, train_loader):
    model.to(device)
    model.train()

    losses = []

    for _, (input, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        #Yolo head is implemented in the loss for training, therefore yolo=False
        output = model(input, yolo=False) #, record_activations=False)
        loss, _ = criterion(output, target)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def validate(model, device, val_loader, num_val_samples=350, roc=True):
    model.to(device)
    model.eval()

    val_precision = []
    val_recall = []

    with torch.inference_mode():
        for idx, (input, target) in tqdm.tqdm(enumerate(val_loader), total=num_val_samples):
            input = input.to(device)
            output = model(input, yolo=True).cpu()
            
            #The right threshold values can be adjusted for the target application
            output = filter_boxes(output, 0.0)
            output = nms(output, 0.5)
            
            precision, recall = precision_recall_levels(target[0], output[0])
            val_precision.append(precision)
            val_recall.append(recall)

            if idx == num_val_samples:
                break
    
    if roc: display_roc(val_precision, val_recall)

    val_ap = ap(val_precision, val_recall)
    return val_ap