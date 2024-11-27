import numpy as np
import os

import torch
from torchvision import datasets
from PIL import Image
from torchvision import transforms

from utils import set_seed

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
               N=10000,
               p=0.8,
               k=9,
               exp="OS",
               force_generation=False,
               seed=0,
               verbose=True,):
    super(CausalMNIST, self).__init__(root, 
                                      transform=None,
                                      target_transform=None)

    self.N = N,
    self.p = p
    self.k = k
    self.exp = exp
    self.seed = seed
    self.force_generation = force_generation
    self.verbose = verbose
    self.prepare_colored_mnist(N=self.N, p=self.p, k=self.k, exp=self.exp, seed=self.seed)
    self.data_label_tuples = torch.load(os.path.join(self.root, 'CausalMNIST', str(k), str(p), str(seed), f'{exp}.pt'))
    self.W = torch.Tensor([obs[1] for obs in self.data_label_tuples])[:,0]
    self.U = torch.Tensor([obs[1] for obs in self.data_label_tuples])[:,1]
    self.T = torch.Tensor([obs[1] for obs in self.data_label_tuples])[:,2]
    self.Y = torch.Tensor([obs[1] for obs in self.data_label_tuples])[:,3]
    self.X = torch.Tensor(np.array([np.array(obs[0]) for obs in self.data_label_tuples]))
    
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

  def prepare_colored_mnist(self, N=10000, p=0.8, k=9, exp='OS', seed=0):
    causal_mnist_dir = os.path.join(self.root, 'CausalMNIST')
    if os.path.exists(os.path.join(causal_mnist_dir, str(k), str(p), str(seed), f'{exp}.pt')) \
        and not self.force_generation:
      if self.verbose: print(f'Causal MNIST dataset already exists (k={k}, p={p}, seed={seed})')
    else:
      if self.verbose: print(f'Generating Causal MNIST (k={k}, p={p}, seed={seed})')
      if not os.path.exists(os.path.join(causal_mnist_dir, str(k), str(p), str(seed))):
        os.makedirs(os.path.join(causal_mnist_dir, str(k), str(p), str(seed)))
      train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
      images = train_mnist.data
      labels = train_mnist.targets

      set_seed(seed)
      dataset = []
      W = np.random.binomial(1, p, N)
      U = np.random.binomial(k, p, N)

      # RCT
      T = np.random.binomial(1, 0.5, N)
      Y = np.round((9*(W/4 + U/(2*k) + T/4) + np.random.binomial(9, 0.5, N))/2).astype(int)
      dataset = []
      for digit in range(10):
          idxs = np.where(Y==digit)[0]
          if len(idxs)==0: 
              continue
          images_digit = images[labels==digit]
          for i, idx in enumerate(idxs):
              x = images_digit[i]
              w = W[idx]
              u = U[idx]
              t = T[idx]
              y = Y[idx]
              x = color_grayscale_arr(np.array(x), background=w, pen=t, pad=4*u)

              dataset.append((x, (w, u, t, y)))

      np.random.shuffle(dataset)
      torch.save(dataset, os.path.join(causal_mnist_dir, str(k), str(p), str(seed), 'RCT.pt'))

      # OS
      T = np.round((np.random.binomial(3, 0.5, N) + W + U/k)/5)
      Y = np.round((9*(W/4 + U/(2*k) + T/4) + np.random.binomial(9, 0.5, N))/2).astype(int)
      dataset = []
      for digit in range(10):
          idxs = np.where(Y==digit)[0]
          if len(idxs)==0: 
              continue
          images_digit = images[labels==digit]
          for i, idx in enumerate(idxs):
              x = images_digit[i]
              w = W[idx]
              u = U[idx]
              t = T[idx]
              y = Y[idx]
              x = color_grayscale_arr(np.array(x), background=w, pen=t, pad=4*u)

              dataset.append((x, (w, u, t, y)))

      np.random.shuffle(dataset)
      torch.save(dataset, os.path.join(causal_mnist_dir, str(k), str(p), str(seed), 'OS.pt'))

def color_grayscale_arr(arr, background=True, pen=True, pad=0):
  '''
  Converts grayscale image changing the background and pen color and zoom.
  
    Args:
        arr: np.array
        background: bool
        pen: bool
        pad: int
    Returns:
        np.array
  '''
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if background: # green
    color = [0, 255, 0]
    if pen: # white
      arr = np.concatenate([arr,
                            255*np.ones((h, w, 1), dtype=dtype),
                            arr], axis=2)
    else: # black
      arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                            255*np.ones((h, w, 1), dtype=dtype)-arr,
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)

  else: # red
    color = [255, 0, 0]
    if pen: # white
      arr = np.concatenate([255*np.ones((h, w, 1), dtype=dtype),
                            arr,
                            arr], axis=2)
    else: # black
      arr = np.concatenate([255*np.ones((h, w, 1), dtype=dtype)-arr,
                            np.zeros((h, w, 1), dtype=dtype),
                            np.zeros((h, w, 1), dtype=dtype)], axis=2)
  if pad>0:
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    arr[:pad, :, :] = color  
    arr[-pad:, :, :] = color
    arr[:, :pad, :] = color
    arr[:, -pad:, :] = color
    arr = Image.fromarray(arr.astype(np.uint8)).resize((28, 28))
  return np.transpose(np.array(arr),(2, 0, 1))