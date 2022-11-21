import os
import torch
from torch import nn
from torchsummary import summary

os.environ['CUDA_VISIBLE_DEVICES']='2'

if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
        nn.Linear(64, 8)
    )
    # net = net.train()
    # summary(net, (3, 256, 256))

    from torchvision.models import resnet18,densenet121
    from thop import profile

    # net = resnet18()
    net = densenet121()
    summary(net, (3, 256, 256))
    model_name = 'cls'
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(net, inputs=(input,), verbose=True)
    print("model: %s | params: %.2f (M)| FLOPs: %.2f (G)" % (
    model_name, params / (1000 ** 2), flops / (1000 ** 3)))  # 这里除以1000的平方，是为了化成M的单位，

    from thop import clever_format
    macs, params = clever_format([flops, params], "%.3f")