import matplotlib.pyplot as plt
import seaborn as sns

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
        plt.rc('font', size=8)
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(f"B={label[0]}; P={label[1]}; Y={label[3]}")  #D={int(label[2])}; Y={label[3]}")  
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    if save: 
        plt.savefig(f'./results/CausalMNIST/{dataset.subsampling}/example.png', bbox_inches='tight')
    plt.show()  


def boxplot_ead(results, subsampling, save=False, path='./results/CausalMNIST/'):
    '''
    Visualize EAD distribution among different models and datasets.

    Args:
        results: pd.DataFrame
        subsampling: str
        save: bool
        path: str
    '''
    EAD_s = results[results['dataset'] == 'train']['EAD'].rename(r'EAD$_{B,Y}^s$')
    EAD_all = results[results['dataset'] == 'all']['EAD'].rename(r'EAD$_{B,Y}$')
    EAD_prob = results[results['dataset'] == 'all']['EAD_prob'].rename(r'EAD$_{B,\hat{Y}}$')
    EAD_binary = results[results['dataset'] == 'all']['EAD_binary'].rename(r'EAD$_{B,\hat{Y}^*}$')

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=[EAD_s, EAD_all, EAD_prob, EAD_binary])
    plt.axhline(y=0.25, color='r', linestyle='--')
    plt.title(f'RCT with {subsampling} subsampling')
    if save:
        plt.savefig(path + f'{subsampling}/boxplot_ead.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()