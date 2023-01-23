# %%
from pathlib import Path

from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import io
from multiprocessing import Pool

# %%
def res(img_path = None):
  if img_path == None:
    return

  image = io.imread(img_path)
  image = resize(image, (512, 512))
  io.imsave(f'./source/data/custom/images_res/{img_path.name}', img_as_ubyte(image))

# %%
Path('./source/data/custom/images_res').mkdir(exist_ok=True, parents=True)
paths = list(Path().glob('./source/data/Melanoma Classification/**/*.jpg'))

done = list(Path().glob('./source/data/custom/images_res/*.jpg'))
done = [x.name for x in done]

missing = []
for path in paths:
  if path.name not in done:
    missing.append(path)

# %%
with Pool(12) as p:
  p.map(res, missing)

# %%
for m in missing:
  res(m)
