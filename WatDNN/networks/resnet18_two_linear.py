import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18TwoLinear(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the original fc layer with two linear layers
        self.base.fc = nn.Sequential(
            nn.Linear(512, 256),  # First Linear
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Second Linear
        )

    def forward(self, x):
        return self.base(x)

    # ResNet18TwoLinear(
    #     (base): ResNet(
    #     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    # (layer1): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer2): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer3): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (layer4): Sequential(
    #     (0): BasicBlock(
    #     (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (downsample): Sequential(
    #     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    # (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (1): BasicBlock(
    #     (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace=True)
    # (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # )
    # )
    # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # (fc): Sequential(
    #     (0): Linear(in_features=512, out_features=256, bias=True)
    # (1): ReLU()
    # (2): Linear(in_features=256, out_features=10, bias=True)
    # )
    # )
    # )
