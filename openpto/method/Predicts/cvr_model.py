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
        # models
        self.pred0 = nn.Linear(num_features, intermediate_size * 4)
        self.pred1 = nn.Linear(intermediate_size * 4, intermediate_size * 2)
        self.pred2 = nn.Linear(intermediate_size * 2, intermediate_size // 2)
        self.pred3 = nn.Linear(intermediate_size // 2, num_targets)

    def forward(self, input_feat):
        num_ins = len(input_feat)
        out_list = list()
        for ins_idx in range(num_ins):
            feat_i = input_feat[ins_idx]
            # user_feat = self.user_emb_layer(u_node)
            learned_feat = F.relu(self.pred0(feat_i))
            learned_feat = F.relu(self.pred1(learned_feat))
            learned_feat = F.relu(self.pred2(learned_feat))
            logits = self.pred3(learned_feat)
            out = torch.sigmoid(logits)
            out_list.append(out)
        return out_list
