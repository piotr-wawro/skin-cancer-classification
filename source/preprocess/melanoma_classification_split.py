# %%
from pathlib import Path

import pandas as pd
import seaborn as sns

# %%
data = pd.read_csv('./source/data/Melanoma Classification/ISIC_2019_Training_GroundTruth.csv')

# %%
labels = pd.DataFrame(columns=['image', 'dx'])

for idx, val in data.iterrows():
  image = val['image']
  dx = val[val == 1].index.item()
  new_row = pd.DataFrame([[image, dx]], columns=['image', 'dx'])

  labels = pd.concat([labels, new_row], axis=0)

labels.reset_index(inplace=True, drop=True)

# %%
sns.histplot(data=labels, x='dx')

# %%
test_dataset = pd.DataFrame(columns=['image', 'dx'])

for dx in labels['dx'].unique():
  test = labels[labels['dx'] == dx].sample(frac=0.3)
  test_dataset = pd.concat([test_dataset, test], axis=0)

train_dataset = labels.drop(test_dataset.index)

Path('./source/data/custom').mkdir(exist_ok=True, parents=True)
test_dataset.to_csv('./source/data/custom/test.csv', index=False)
train_dataset.to_csv('./source/data/custom/train.csv', index=False)
