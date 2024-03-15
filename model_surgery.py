from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from mobilenetv2 import MobileNetV2

BATCH_SIZE = 32
LEARNING_RATE_CLASSIFIER = 0.001
LEARNING_RATE_EPS = 0.01
DATA_DIR = "/home/tejasj/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_ITERS = 10_000


class SurgerizedModel(MobileNetV2):
    def __init__(self, num_basis: int, num_classes=10, width_mult=1.0):
        super(SurgerizedModel, self).__init__(num_classes, width_mult)

        self.num_basis = num_basis

        self.classifier = None
        self.new_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self.eps = nn.Parameter(torch.rand(num_basis, requires_grad=True))

    def update_weights(self, basis: Dict[str, torch.Tensor]) -> None:
        new_state_dict = self.state_dict()
        for key, value in basis.items():
            if key in self.state_dict():
                new_state_dict[key] = torch.einsum(
                    "l...,l->...", value, self.eps.detach().cpu()
                )
        self.load_state_dict(new_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.new_classifier(x)
        return x


if __name__ == "__main__":
    # Create a summary writer
    writer = SummaryWriter(
        log_dir="/home/tejasj/data/neural_fisher_kernel/runs/surgery"
    )

    transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ]
    )
    dataset = CIFAR10(root=DATA_DIR, train=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )

    model = SurgerizedModel(num_basis=20)

    optimizer_classifier = torch.optim.AdamW(
        model.new_classifier.parameters(), lr=LEARNING_RATE_CLASSIFIER
    )

    basis = torch.load(
        "checkpoints/mobilenet_v2_l_20/basis_40000.pt", map_location="cpu"
    )
    model.update_weights(basis)
    model.to(DEVICE)

    step = 0
    while True:
        for i, samples in tqdm(enumerate(dataloader)):
            inputs = samples[0].to(DEVICE)
            labels = samples[1].to(DEVICE)

            out = model(inputs)
            loss = F.cross_entropy(out, labels)
            loss.backward()

            eps_grad = 0
            for key, param in model.named_parameters():
                if key in basis:
                    eps_grad += torch.einsum(
                        "l...,...->l", basis[key], param.grad.cpu()
                    )

            model.eps.data = model.eps.data - LEARNING_RATE_EPS * eps_grad.to(
                model.eps.device
            )
            optimizer_classifier.step()

            model.update_weights(basis)

            if step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar(
                    "train/accuracy",
                    (out.argmax(1) == labels).float().mean().item(),
                    step,
                )

            if step > TOTAL_ITERS:
                print("Finished training!")
                exit(0)

            step += 1
