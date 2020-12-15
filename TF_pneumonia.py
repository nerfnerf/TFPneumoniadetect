import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = pathlib.Path("/run/user/1000/gvfs/smb-share:server=tower.local,share=coding/chest_xray/chest_xray/train")

train_normal = list(data_dir.glob('train/NORMAL/*.jpeg'))


#image = PIL.Image.open(str(train_normal[0]))
#image.show()

amount = len(train_normal)

print(amount)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)