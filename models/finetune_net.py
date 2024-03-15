import torch
import torch.nn as nn
import copy
import functools

from models.energy_net import MODEL_MAPPING
from models.mobilenetv2 import MobileNetV2

def create_finetune_net(base_model):
    class FinetuneNet(base_model):
        def __init__(self, configs, pretrained_param, basis_param):
            super(FinetuneNet, self).__init__(configs.num_classes, configs.width_mult)
            self.num_basis = configs.basis_size
            self.pretrained_param = pretrained_param
            self.basis_param = basis_param
            self.classifier = None
            self.new_classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, configs.num_classes),
            )
            self.eps = nn.Parameter(torch.rand(self.num_basis, requires_grad=True))

        def cache_state_dict(self):
            if hasattr(self, "cached_state_dict"):
                raise ValueError("Cached state dict already exists.")

            self.cached_state_dict = [
                copy.deepcopy(
                    dict(
                        map(
                            functools.partial(move_to_device, device="cpu"),
                            self.pretrained_param().items(),
                        )
                    )
                )
            ]
        def update_weights(self) -> None:
            if not hasattr(self, "cached_state_dict"):
                raise ValueError(
                    "Cached state dict not found. Call cache_state_dict first."
                )
            cached_state_dict = self.cached_state_dict[0]
            new_state_dict = self.state_dict()
            for key, value in self.basis_param.items():
                if key in self.state_dict():
                    new_state_dict[key] = cached_state_dict[
                                              key
                                          ].detach().cpu() + torch.einsum(
                        "l...,l->...", value, self.eps.detach().cpu()
                    )
            # Store the current state dict to use in the next iteration
            self.load_state_dict(new_state_dict)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.mean([2, 3])
            x = self.new_classifier(x)
            return x

    return FinetuneNet


def finetunenet(configs, model_path, basis_path, device):
    pretrained_param = torch.load(model_path, map_location="cpu")
    basis_param = torch.load(basis_path, map_location="cpu")
    FinetuneNet = create_finetune_net(MODEL_MAPPING[configs.model_name])
    model = FinetuneNet(configs, pretrained_param, basis_param)
    return model
