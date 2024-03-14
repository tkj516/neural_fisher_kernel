import torch
import torch.nn as nn
from models.energy_net import MODEL_MAPPING
from models.mobilenetv2 import MobileNetV2

class Finetune_Net(nn.Module):
    def __init__(self, configs, pretrained_param, basis_param, **kwargs):
        super(Finetune_Net, self).__init__()
        self.base_model = MODEL_MAPPING[configs.model_name](**kwargs)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.pretrained_param = pretrained_param
        self.basis_param = basis_param
        self.principle_coeff = nn.Linear(configs.basis_size, 1, bias=False)

    def forward(self, x):
        for key in self.basis_param.keys():  # Iterate over the keys of the parameter subsets
            updated_param = self.pretrained_param[key] + self.principle_coeff(self.basis_param[key].T).squeeze(-1)
            self.base_model.state_dict()[key].data.copy_(updated_param)

        return self.base_model(x)

def finetunenet(configs, model_path, basis_path, device, **kwargs):
    pretrained_param = torch.load(model_path, map_location=device)
    basis_param = torch.load(basis_path, map_location=device)
    model = Finetune_Net(configs, pretrained_param, basis_param, **kwargs)
    return model
