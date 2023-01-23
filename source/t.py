# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch

# %%
train = pd.read_csv('./source/data/custom/train.csv')
test = pd.read_csv('./source/data/custom/test.csv')

# %%
test.groupby('dx').count()

# %%
len(test)+len(train)

# %%
loss_fn = nn.CrossEntropyLoss()

loss_fn(torch.tensor([0,10,10.]), torch.tensor(2))

# %%
torch.tensor([0,10,10.]).argmax(1)==1
# %%
