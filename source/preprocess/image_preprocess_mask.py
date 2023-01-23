# %%
from pathlib import Path

from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.morphology import closing, disk, opening, convex_hull_object, dilation
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import io
import numpy as np
from numpy import newaxis
from multiprocessing import Pool
import matplotlib.pyplot as plt

# %%
def segment(img_path = None, plot = False):
  # img_path = './source/data/Melanoma Classification/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0014117_downsampled.jpg'
  if img_path == None:
    return

  image = io.imread(img_path)

  if plot:
    plt.figure(figsize=(6,6), dpi=160)
    plt.subplots_adjust(hspace=0.3)
    ax = plt.subplot(4,3,1)
    ax.set_title('original', fontsize=8)
    ax.set_axis_off()
    ax.imshow(image)

  image = resize(image, (512, 512))
  gray_image = rgb2gray(image)

  segmented_image = KMeans(n_clusters=2, n_init=1, random_state=0)\
                      .fit(gray_image.reshape(-1,1))\
                      .labels_\
                      .reshape(gray_image.shape)

  if plot:
    ax = plt.subplot(4,3,2)
    ax.set_title('segmented v1', fontsize=8)
    ax.set_axis_off()
    ax.imshow(segmented_image, cmap='gray')

  cluster0 = gray_image[segmented_image == 0].mean()
  cluster1 = gray_image[segmented_image == 1].mean()

  resegment = False
  if cluster0 < 0.1:
    if cluster0 < cluster1:
      gray_image[segmented_image == 0] = cluster1
    else:
      gray_image[segmented_image == 1] = cluster0
    resegment = True
  elif cluster1 < 0.1:
    if cluster1 < cluster0:
      gray_image[segmented_image == 1] = cluster0
    else:
      gray_image[segmented_image == 0] = cluster1
    resegment = True

  if resegment:
    segmented_image = KMeans(n_clusters=2, n_init=1, random_state=0)\
                        .fit(gray_image.reshape(-1,1))\
                        .labels_\
                        .reshape(gray_image.shape)

    cluster0 = gray_image[segmented_image == 0].mean()
    cluster1 = gray_image[segmented_image == 1].mean()

    if plot:
      ax = plt.subplot(4,3,3)
      ax.set_title('segmented v2', fontsize=8)
      ax.set_axis_off()
      ax.imshow(segmented_image, cmap='gray')

  if cluster0 < cluster1:
    if cluster0 > 0.1:
      naevus = 0
    else:
      naevus = 1
  elif cluster0 > cluster1:
    if cluster1 > 0.1:
      naevus = 1
    else:
      naevus = 0

  mask = np.zeros_like(gray_image)
  mask[segmented_image == naevus] = 1
  mask[segmented_image != naevus] = 0

  mask_closing = closing(mask, footprint = disk(20))
  mask_opening = opening(mask_closing, footprint = disk(20))
  mask_convex = convex_hull_object(mask_opening)
  mask_dilation = dilation(mask_convex, footprint = disk(40))
  mask_f = mask_dilation[:,:,newaxis]

  if plot:
    ax = plt.subplot(4,3,4)
    ax.set_title('mask', fontsize=8)
    ax.set_axis_off()
    ax.imshow(mask, cmap='gray')

    ax = plt.subplot(4,3,5)
    ax.set_title('mask_closing', fontsize=8)
    ax.set_axis_off()
    ax.imshow(mask_closing, cmap='gray')

    ax = plt.subplot(4,3,6)
    ax.set_title('mask_opening', fontsize=8)
    ax.set_axis_off()
    ax.imshow(mask_opening, cmap='gray')

    ax = plt.subplot(4,3,7)
    ax.set_title('mask_convex', fontsize=8)
    ax.set_axis_off()
    ax.imshow(mask_convex, cmap='gray')

    ax = plt.subplot(4,3,8)
    ax.set_title('mask_dilation', fontsize=8)
    ax.set_axis_off()
    ax.imshow(mask_dilation, cmap='gray')

  new_image = image.copy()
  new_image *= mask_f

  if plot:
    ax = plt.subplot(4,3,9)
    ax.set_title('end', fontsize=8)
    ax.set_axis_off()
    ax.imshow(new_image)

  io.imsave(f'./source/data/custom/images/{img_path.name}', img_as_ubyte(new_image))

# segment(plot = True)

# %%
Path('./source/data/custom/images').mkdir(exist_ok=True, parents=True)
paths = list(Path().glob('./source/data/Melanoma Classification/**/*.jpg'))

done = list(Path().glob('./source/data/custom/images/*.jpg'))
done = [x.name for x in done]

missing = []
for path in paths:
  if path.name not in done:
    missing.append(path)

# %%
with Pool(12) as p:
  print(p.map(segment, missing))

# %%
for m in missing:
  segment(m, plot=True)
