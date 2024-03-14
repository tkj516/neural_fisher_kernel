import torch
import torch.nn.functional as F
from models.mobilenetv2 import MobileNetV2


class EnergyNet(MobileNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns \sum_{y \in Y} p(y | x) f^y(x).
        """
        out = super().forward(x)
        probs = F.softmax(out, dim=-1)
        return torch.sum(probs.detach() * out, dim=-1)

def energynet(model_path, pretrained=False, device="cpu", **kwargs):
    model = EnergyNet(**kwargs)
    if pretrained:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    return model