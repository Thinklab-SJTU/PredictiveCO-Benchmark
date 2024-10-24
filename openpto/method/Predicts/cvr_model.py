import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        intermediate_size,
        output_activation,
        **args,
    ):
        super(CVRModel, self).__init__()

    def forward(self, input_feat):
        return
