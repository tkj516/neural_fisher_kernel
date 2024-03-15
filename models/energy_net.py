import torch
import torch.nn.functional as F
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2

MODEL_MAPPING = {
'MobileNet': MobileNetV2,
# define other model mapping here
}

def create_energy_net(base_model):
    class EnergyNet(base_model):
        def __init__(self, **kwargs):
            super(EnergyNet, self).__init__(**kwargs)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = super(EnergyNet, self).forward(x)  # Use the forward method of the base model
            probs = torch.nn.functional.softmax(out, dim=-1)
            return torch.sum(probs.detach() * out, dim=-1)

    return EnergyNet

def energynet(model_name, model_path=None, pretrained=False, device="cpu", **kwargs):
    EnergyNet = create_energy_net(MODEL_MAPPING[model_name])
    model = EnergyNet(**kwargs)
    if pretrained and model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    return model
