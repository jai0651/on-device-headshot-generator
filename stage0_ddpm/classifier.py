"""A small MNIST classifier: the instrument that checks "did it draw the digit I asked for".

    uv run python -m stage0_ddpm.classifier          # ~1 min on CPU, ~99% test accuracy

Same role as the face recognizer in stage 2: a separately trained judge, never used to generate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10))

    def forward(self, x):  # x in [-1, 1], like the diffusion model's images
        return self.net(x)


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf), batch_size=256, shuffle=True)
    test = DataLoader(datasets.MNIST("data", train=False, download=True, transform=tf), batch_size=1000)
    model = Classifier()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(2):
        model.train()
        for x, y in train:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = sum((model(x).argmax(1) == y).sum().item() for x, y in test) / 10000
        print(f"epoch {epoch}: test accuracy {acc:.2%}")
    torch.save(model.state_dict(), "runs/stage0/mnist_classifier.pt")
    print("-> runs/stage0/mnist_classifier.pt")


if __name__ == "__main__":
    main()
