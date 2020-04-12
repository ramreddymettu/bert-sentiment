import config

import torch
import torch.nn as nn

from tqdm import tqdm

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train(data_loader, model, optimizer, device, scheduler):
    #model.to(device)
    model.train()

    for bi, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = batch_data['ids']
        masks = batch_data['mask']
        token_type_ids = batch_data['token_type_ids']
        targets = batch_data['target']

        ids = ids.to(device, torch.long)
        masks = masks.to(device, torch.long)
        token_type_ids = token_type_ids.to(device, torch.long)
        targets = targets.to(device, torch.float)
        
        optimizer.zero_grad()
        outputs = model(
            ids,
            masks,
            token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()


def eval(data_loader, model, device):
    #model.to(device)
    model.eval()

    preds_final = []
    targets_final = []

    for bi, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = batch_data['ids']
        masks = batch_data['mask']
        token_type_ids = batch_data['token_type_ids']
        targets = batch_data['target']

        ids = ids.to(device, torch.long)
        masks = masks.to(device, torch.long)
        token_type_ids = token_type_ids.to(device, torch.long)
        targets = targets.to(device, torch.long)

        outputs = model(
            ids,
            masks,
            token_type_ids
        )

        pred = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        targets = targets.cpu().detach().numpy().tolist()

        preds_final.extend(pred)
        targets_final.extend(targets)

    return targets_final, preds_final


