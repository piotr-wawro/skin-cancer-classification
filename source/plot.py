# %%
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# %%
train = pd.read_csv('./train.txt', names=['train loss', 'train acc'])
test = pd.read_csv('./test.txt', names=['test loss', 'test acc'])

# %%
data = pd.concat([train, test], axis=1)

# %%
def obj(x, a, b):
  return np.sqrt(x*a)+b

x = data.index.to_numpy()
y = data.loc[:, 'train acc']
popt, _ = curve_fit(obj, x, y)
pred = obj(np.arange(1,400,1), *popt)

# %%
fig, ax = plt.subplots(2, 1)
plt.subplot(2,1,1)
sns.lineplot(data.loc[:, ['train acc', 'test acc']], legend=False, errorbar=None)
plt.ylabel('acc')

plt.subplot(2,1,2)
sns.lineplot(data.loc[:, ['train loss', 'test loss']], legend=False, errorbar=None)
plt.ylabel('loss')

plt.xlabel('epoch')
plt.legend(labels=["train", "test"])
plt.show()

# %%
plt.savefig('lr 1e-2')
