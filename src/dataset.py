import numpy as np
import os

import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image

class CausalMNIST(datasets.VisionDataset):
  """
  Causal MNIST dataset for testing Treatment Effects Estimation 
  algorithms on higher dimensional data.

    Args:
        root (string): Data root directory (default='./data').
        env (string): The dataset environment to load. Options are 
            'train', 'val', 'test', 'train_full', and 'all' 
            (default='all').
        transform: A function/transform that  takes in an PIL image
            and returns a transformed version; e.g., 
            'transforms.RandomCrop' (default=None).
        target_transform (callable, optional): A function/transform 
            that takes in the target and transforms it (default=None).
        force_generation (bool): If True, forces the generation of the 
            dataset (default=False).
        force_split (bool): If True, forces the split of the dataset 
            into train, val, and test (default=False).
        subsampling (string): The subsampling method to use. Options 
            are 'random' and 'biased' (default='random').
        verbose (bool): If True, prints the dataset generation and
            split progress (default=True).
  """
  def __init__(self, 
               root='./data', 
               env='all', 
               transform=None, 
               target_transform=None, 
               force_generation=False,
               force_split=False,
               subsampling="random",
               verbose=True):
    super(CausalMNIST, self).__init__(root, 
                                      transform=transform,
                                      target_transform=target_transform)
    self.force_generation = force_generation
    if force_generation:
      force_split = True
    self.force_split = force_split
    self.subsampling = subsampling
    self.verbose = verbose

    self.prepare_colored_mnist()
    if env in ['train', 'val', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, env) + '.pt')
    elif env == 'train_full':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, 'train.pt')) + \
                               torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, 'val.pt')) 
    elif env == 'all':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, 'train.pt')) + \
                               torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, 'val.pt')) + \
                               torch.load(os.path.join(self.root, 'CausalMNIST', self.subsampling, 'test.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are supervised, unsupervised, and all.')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    causal_mnist_dir = os.path.join(self.root, 'CausalMNIST')
    if os.path.exists(os.path.join(causal_mnist_dir, 'dataset.pt')) \
        and not self.force_generation:
      if self.verbose: print('Causal MNIST dataset already exists')
    else:
      if self.verbose: print('Generating Causal MNIST')
      train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

      dataset = []

      n = 0
      p_b = 0.5

      images = train_mnist.data
      labels = train_mnist.targets
      # split images in 10 datasets (1 per label from 0 to 9)
      images_ordered = []
      for label in range(10):
        images_ordered.append(images[labels == label])

      flag = True
      while flag: 
        n_b = np.random.binomial(1, p_b)
        n_p = np.random.uniform(0, 2)
        n_d = np.random.uniform(0, 10)

        B = n_b # (0=red, 1=green)
        if B:
          D = int(np.sqrt(n_d*10))
        else:
          D = int(n_d)
        P = 1 if (B + D/9 - n_p)>0 else 0 # (0=black, 1=white) 
        Y = 1 if D > 4 else 0

        image = images_ordered[D][0]
        images_ordered[D] = images_ordered[D][1:]
        X = color_grayscale_arr(np.array(image), background=B, pen=P)
        if len(images_ordered[D]) == 0:
          flag = False
        
        dataset.append((Image.fromarray(X), (B, P, D, Y)))

      if not os.path.exists(causal_mnist_dir):
        os.makedirs(causal_mnist_dir)
      torch.save(dataset, os.path.join(causal_mnist_dir, 'dataset.pt'))

    if os.path.exists(os.path.join(causal_mnist_dir, self.subsampling, 'train.pt')) \
        and os.path.exists(os.path.join(causal_mnist_dir, self.subsampling, 'val.pt')) \
        and os.path.exists(os.path.join(causal_mnist_dir, self.subsampling, 'test.pt')) \
        and not self.force_split:
      if self.verbose: print('Causal MNIST dataset environments already exists')
    else: 
      if self.verbose: print('Splitting Causal MNIST into train, val and test sets')
      dataset = torch.load(os.path.join(causal_mnist_dir, 'dataset.pt'))

      np.random.shuffle(dataset)
      n = len(dataset)
      n_train = int(0.15 * n)
      n_val = int(0.05 * n)
      
      if self.subsampling=="random":
        train = dataset[:n_train]
        val = dataset[n_train:n_train+n_val]
        test = dataset[n_train+n_val:]
      elif self.subsampling=="biased":
        train = []
        val = []
        test = []
        n_train_temp_p0 = 0
        n_train_temp_p1 = 0
        n_val_temp_p0 = 0
        n_val_temp_p1 = 0
        for i in range(n):
          if dataset[i][1][1] == 1:
            if n_train_temp_p1 < n_train*0.9:
              train.append(dataset[i])
              n_train_temp_p1 += 1
            elif n_val_temp_p1 < n_val*0.8:
              val.append(dataset[i])
              n_val_temp_p1 += 1
            else:
              test.append(dataset[i])
          else:
            if n_train_temp_p0 < n_train*0.1:
              train.append(dataset[i])
              n_train_temp_p0 += 1
            elif n_val_temp_p0 < n_val*0.2:
              val.append(dataset[i])
              n_val_temp_p0 += 1
            else:
              test.append(dataset[i])
          # n_train_temp = 0
          # n_val_temp = 0
          # n_temp = 0
          # for i in range(n):
          # if  np.random.binomial(1, 0.5):
          #   if dataset[i][1][0] == 1 and n_train_temp < n_train:
          #     if dataset[i][1][1] == 0:
          #       train.append(dataset[i])
          #       n_train_temp += 1
          #     else:
          #       test.append(dataset[i])
          #   else:
          #     if dataset[i][1][1] == 1 and n_val_temp < n_val:
          #       val.append(dataset[i])
          #       n_val_temp += 1
          #     else:
          #       test.append(dataset[i])
          # else: 
          #   if dataset[i][1][0] == 0 and n_train_temp < n_train:
          #     if dataset[i][1][1] == 1:
          #       train.append(dataset[i])
          #       n_train_temp += 1
          #     else:
          #       test.append(dataset[i])
          #   else:
          #     if dataset[i][1][1] == 0 and n_val_temp < n_val:
          #       val.append(dataset[i])
          #       n_val_temp += 1
          #     else:
          #       test.append(dataset[i])
      else:
        raise ValueError("Subsampling method not recognized")
      
      causal_mnist_dir = os.path.join(causal_mnist_dir, self.subsampling)
      if not os.path.exists(causal_mnist_dir):
          os.makedirs(causal_mnist_dir)
      torch.save(train, os.path.join(causal_mnist_dir, 'train.pt'))
      torch.save(val, os.path.join(causal_mnist_dir, 'val.pt'))
      torch.save(test, os.path.join(causal_mnist_dir, 'test.pt'))

def color_grayscale_arr(arr, background=True, pen=True):
  '''
  Converts grayscale image changing the background and pen color.
  
    Args:
        arr: np.array
        background: bool
        pen: bool
    Returns:
        np.array
  '''
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if background: # green
    if pen: # white
      arr = np.concatenate([arr,
                            255*np.ones((h, w, 1), dtype=dtype),
                            arr], axis=2)
    else: # black
      arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                            255*np.ones((h, w, 1), dtype=dtype)-arr,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)

  else: # red
    if pen: # white
      arr = np.concatenate([255*np.ones((h, w, 1), dtype=dtype),
                            arr,
                            arr], axis=2)
    else: # black
      arr = np.concatenate([255*np.ones((h, w, 1), dtype=dtype)-arr,
                            np.zeros((h, w, 1), dtype=dtype),
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr
