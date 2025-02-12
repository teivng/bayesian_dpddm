import torch

def eval_acc(preds, y):
  map_preds = torch.argmax(preds, dim=1)
  return (map_preds == y).float().mean()