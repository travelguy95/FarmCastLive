import scipy
from scipy.io import savemat
from torchvision import datasets, transforms
import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

data_list = []
for start_week in range(0,1165):
  running_images = np.load("/content/drive/MyDrive/precipitation/"+str(start_week)+".npy")
  for weeks in range(0,63):
    next_image = np.load("/content/drive/MyDrive/precipitation/"+str(start_week+weeks+1)+".npy")
    running_images = np.dstack((running_images,next_image))
  data_list.append(running_images)

train_data = {'u': data_list}
train_loader = DataLoader(train_data)
train_dataset = train_loader.dataset
savemat('/content/drive/MyDrive/precipitation/precipitation.mat', train_dataset)
