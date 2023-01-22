# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch

# %%
train = pd.read_csv('./data/custom/train.csv')
test = pd.read_csv('./data/custom/test.csv')

# %%
train['dx'].unique()

# %%
len(test)+len(train)

# %%
loss_fn = nn.CrossEntropyLoss()

loss_fn(torch.tensor([0,10,10.]), torch.tensor(2))

# %%
torch.tensor([0,10,10.]).argmax(1)==1
# %%
