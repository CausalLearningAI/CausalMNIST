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
               train_ratio=0.05, 
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
    self.train_ratio = train_ratio

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

      images = train_mnist.data
      labels = train_mnist.targets
      dataset = []

      p_B = 0.5 # if change, fix a,b,c,d according to the law of total probability
      p_P = 0.5 # if change, fix a,b,c,d according to the law of total probability
      threshold = 3
      p_Y = (9-threshold)/10
      a = p_Y + 0.2
      b = p_Y - 0.2
      c = p_Y + 0.1
      d = p_Y - 0.1
      self.ATE = (a-b)*p_P + (c-d)*(1-p_P)

      for (image, label) in zip(images, labels):
        D  = label
        Y = 1 if D > threshold else 0
        if Y:
          p = [a*(p_B*p_P)/p_Y, b*((1-p_B)*p_P)/p_Y, c*(p_B*(1-p_P))/p_Y, d*((1-p_B)*(1-p_P))/p_Y]
          aux = np.random.choice([1, 2, 3, 4], p=p)
          if aux==1:
            B = 1
            P = 1
          elif aux==2:
            B = 0
            P = 1
          elif aux==3:
            B = 1
            P = 0
          else:
            B = 0
            P = 0
        else:
          p = [(1-a)*(p_B*p_P)/(1-p_Y), (1-b)*((1-p_B)*p_P)/(1-p_Y), (1-c)*(p_B*(1-p_P))/(1-p_Y), (1-d)*((1-p_B)*(1-p_P))/(1-p_Y)]
          aux = np.random.choice([1, 2, 3, 4], p=p)
          if aux==1:
            B = 1
            P = 1
          elif aux==2:
            B = 0
            P = 1
          elif aux==3:
            B = 1
            P = 0
          else:
            B = 0
            P = 0
        X = color_grayscale_arr(np.array(image), background=B, pen=P)
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
      n_train = int(self.train_ratio * n)
      n_val = int(self.train_ratio * n)
      
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
          if dataset[i][1][1] == 0:
            if n_train_temp_p1 < n_train*1:
              train.append(dataset[i])
              n_train_temp_p1 += 1
            elif n_val_temp_p1 < n_val*0.5:
              val.append(dataset[i])
              n_val_temp_p1 += 1
            else:
              test.append(dataset[i])
          else:
            if n_train_temp_p0 < n_train*0:
              train.append(dataset[i])
              n_train_temp_p0 += 1
            elif n_val_temp_p0 < n_val*0.5:
              val.append(dataset[i])
              n_val_temp_p0 += 1
            else:
              test.append(dataset[i])
        np.random.shuffle(train)
        np.random.shuffle(val)
        np.random.shuffle(test)
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
