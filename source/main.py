# %%
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import Lambda
from tqdm import tqdm
import matplotlib.pyplot as plt

from DatasetCustom import DatasetCustom
from DCNN import DCNN

# %%
dataset = DatasetCustom(
  annotation_file = './source/data/custom/train.csv',
  img_dir = './source/data/custom/images',
  transform = torch.nn.Sequential(
    torchvision.transforms.ConvertImageDtype(torch.float32),
  ),
  target_transform = Lambda(lambda x: torch.tensor(x, dtype=torch.long))
)
train = DataLoader(dataset, 16)

# %%
dataset = DatasetCustom(
  annotation_file = './source/data/custom/test.csv',
  img_dir = './source/data/custom/images',
  transform = torch.nn.Sequential(
    torchvision.transforms.ConvertImageDtype(torch.float32),
  ),
  target_transform = Lambda(lambda x: torch.tensor(x, dtype=torch.long))
)
test = DataLoader(dataset, 16)

# %%
# for image, label in test:
#   for i in range(4):
#     img = torch.permute(image[i], (1,2,0))
#     plt.imshow(img)
#     plt.title(label[i])
#     plt.show()
#   break

# %%
def train_loop(dataloader, model):
  loss = 0.0
  acc = 0.0

  model.train()
  for x, y in tqdm(dataloader, position=1, leave=False):
    x = x.to(model.device)
    y = y.to(model.device)

    pred = model(x)
    l = loss_fn(pred, y)
    optim.zero_grad()
    l.backward()
    optim.step()

    loss += l.item()
    acc += (pred.argmax(1) == y).sum().item()

  loss /= len(dataloader)
  acc /= len(dataloader.dataset)

  with Path('./train.txt').open('a+') as file:
    file.write(f"{loss}, {acc}\n")

def test_loop(dataloader, model):
  loss = 0.0
  acc = 0.0

  model.eval()
  with torch.no_grad():
    for x, y in tqdm(dataloader, position=1, leave=False):
      x = x.to(model.device)
      y = y.to(model.device)

      pred = model(x)
      l = loss_fn(pred, y)

      loss += l.item()
      acc += (pred.argmax(1) == y).sum().item()

  loss /= len(dataloader)
  acc /= len(dataloader.dataset)

  with Path('./test.txt').open('a+') as file:
    file.write(f"{loss}, {acc}\n")

# %%
model_name = 'DCNN.pth'
model = DCNN()
optim = torch.optim.Adadelta(model.parameters(), lr=1e-2)

max_count = train.dataset.annotations['dx'].value_counts().max()
weights = [
  max_count/train.dataset.annotations['dx'].value_counts()[0] * 2,
  max_count/train.dataset.annotations['dx'].value_counts()[1],
]
loss_fn = nn.CrossEntropyLoss(torch.tensor(weights, dtype=torch.float).to(model.device))

# %%
try:
  checkpoint = torch.load(model_name)
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  model.eval()
  print('Model restored')
except:
  pass

# %%
epochs = 100
for t in tqdm(range(epochs), position=0):
    train_loop(train, model)
    test_loop(test, model)

    if (t+1) % 5 == 0:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
      }, model_name)
print("Done!")
