import torch
x = torch.randn(1, 3, 32, 32).cuda()
layer = torch.nn.Conv2d(3, 64, kernel_size=5).cuda()
y = layer(x)
print("Output shape:", y.shape)