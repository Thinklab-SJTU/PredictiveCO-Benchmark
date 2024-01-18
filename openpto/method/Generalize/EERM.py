import torch
import torch.nn as nn
import torch.nn.functional as F

class EERM(nn.Module):
    def __init__(self, pred_model):
        super(Model, self).__init__()
        self.pred_model = pred_model
    
    