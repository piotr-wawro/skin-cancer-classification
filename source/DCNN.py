import torch
from torch import nn
from torchinfo import summary

class DCNN(nn.Module):
  def __init__(self):
    super(DCNN, self).__init__()
    self.conv0 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),

      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(12),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(12),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(12),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(12),
      nn.LeakyReLU(),

      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),

      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(24),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(24),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(24),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(24),
      nn.LeakyReLU(),

      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),

      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Flatten(),
    )

    self.linear = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(7200, 3100),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(3100, 3100),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(3100, 3100),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(3100, 2),
    )

    if torch.cuda.is_available():
      self.cuda()
    else:
      self.cpu()
    self.device = next(self.parameters()).device

  def forward(self, x):
    out0 = self.conv0(x)
    out1 = self.linear(out0)
    return out1

if __name__ == '__main__':
  model = DCNN()
  summary(
    model=model,
    input_size=(16, 3, 512, 512),
  )
