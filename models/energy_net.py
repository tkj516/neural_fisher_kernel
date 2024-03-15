import torch
import torch.nn.functional as F
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2

MODEL_MAPPING = {
'MobileNet': MobileNetV2,
# define other model mapping here
}
'''
class EnergyNet(nn.Module):
    def __init__(self, model_name, **kwargs):
        super(EnergyNet, self).__init__()
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Unknown model {model_name}")
        self.base_model = MODEL_MAPPING[model_name](**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns \sum_{y \in Y} p(y | x) f^y(x).
        """
        out = self.base_model.forward(x)
        probs = F.softmax(out, dim=-1)
        return torch.sum(probs.detach() * out, dim=-1)
'''
class EnergyNet(nn.Module):
    def __init__(self, base_model = MobileNetV2, **kwargs):
        base_model.__init__(self, **kwargs)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns \sum_{y \in Y} p(y | x) f^y(x).
        """
        out = base_model.forward(x)
        probs = F.softmax(out, dim=-1)
        return torch.sum(probs.detach() * out, dim=-1)

def energynet(model_name, model_path, pretrained=False, device="cpu", **kwargs):
    model = EnergyNet(MODEL_MAPPING[model_name], **kwargs)
    if pretrained:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    return model