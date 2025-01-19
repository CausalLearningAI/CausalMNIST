import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualize_n_digits(dataset, n=36, save=False):
    '''
    Visualize first n images from the dataset.

    Args:
        dataset: torch.utils.data.Dataset
        n: int
        save: bool
    '''
    if n < 6:
        columns = n
    else:
        columns = 6   
    rows = n//6 + 1
    fig = plt.figure(figsize=(13, rows*2.7))
    ax = []
    for i in range(n):
        img, label = dataset[i]
        img = np.transpose(img,(1,2,0))
        plt.rc('font', size=8)
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(f"W={label[0]}; U={label[1]}; T={label[2]}; Y={label[3]}")  
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    if save: 
        if not os.path.exists(f'./results/{dataset.e}/{dataset.pW}/{dataset.pU}'):
            os.makedirs(f'./results/{dataset.e}/{dataset.pW}/{dataset.pU}')
        plt.savefig(f'./results/{dataset.e}/{dataset.pW}/{dataset.pU}/{dataset.exp}.png', bbox_inches='tight')
    plt.show()  

