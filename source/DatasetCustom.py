from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd


class DatasetCustom(Dataset):
  def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
    self.annotations = pd.read_csv(annotation_file)
    self.annotations['dx'] = self.annotations['dx'].map({
      'MEL': 0, # melanoma | czerniak
      'NV': 1, # melanocytic nevi | znamiona melanocytowe
      'BCC': 0, # basal cell carcinoma | rak podstawnokomórkowy
      'AK': 1, # actinic keratoses | rogowacenie słoneczne
      'BKL': 1, # benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) | łagodne zmiany przypominające rogowacenie (plama soczewicowata / rogowacenie łojotokowe i rogowacenie podobne do liszaja płaskiego)
      'DF': 1, # dermatofibroma | dermatofibroma
      'VASC': 1, # vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage) | zmiany naczyniowe (naczyniaki, angiokeratomy, ziarniniaki i krwotoki ropotwórcze)
      'SCC': 0, # Squamous cell carcinoma | rak kolczystokomórkowy
    })
    # 0 - dangerous
    # 1 - safe

    # dangerous = self.annotations[self.annotations['dx'] == 0].sample(frac=1)
    # safe = self.annotations[self.annotations['dx'] == 1].sample(n=len(dangerous))
    # self.annotations = pd.concat([dangerous, safe], axis=0)
    # self.annotations = self.annotations.reset_index(drop=True)

    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    img_name = self.annotations.loc[idx, 'image'] + '.jpg'
    img = read_image(self.img_dir + '/' + img_name)
    label = self.annotations.loc[idx, 'dx']

    if self.transform:
      img = self.transform(img)
    if self.target_transform:
      label = self.target_transform(label)

    return img, label
