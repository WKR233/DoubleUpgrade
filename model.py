import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 54)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value