import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from dataset import CausalMNIST
from models import ConvNet

def performances_val(model, data_loader, verbose=False, device = 'cpu'):
    model.eval()
    with torch.no_grad():
        images = torch.stack([image for image, _ in data_loader.dataset])
        output = model(images)
        y_pred_probs = torch.sigmoid(output)
        y_pred_bin = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                             torch.Tensor([1.0]).to(device),
                             torch.Tensor([0.0]).to(device))  
        labels = torch.tensor([label for _, label in data_loader.dataset])
        y = labels[:,3].float()
        t = labels[:,0]
        ead = torch.mean(y[t==1]) - torch.mean(y[t==0])
        ead_prob = torch.mean(y_pred_probs[t==1]) - torch.mean(y_pred_probs[t==0])
        #ead_binary = torch.mean(y_pred_bin[t==1]) - torch.mean(y_pred_bin[t==0])
        TEB = ead_prob - ead
        loss = F.binary_cross_entropy_with_logits(output, y)
        acc = accuracy_score(y, y_pred_bin)
        bacc = balanced_accuracy_score(y, y_pred_bin)
        
        if verbose:
            print(f'Validation set: Average loss: {loss.item():.4f}, Accuracy: {acc:.2f}, Balanced Accuracy: {bacc:.2f}, TEB: {TEB:.2f}')
    return loss.item(), acc.item(), bacc.item(), TEB.item()

def performances_all(model, data_loader, verbose=False, device = 'cpu'):  
    model.eval()
    with torch.no_grad():
        images = torch.stack([image for image, _ in data_loader.dataset])
        output = model(images)
        y_pred_probs = torch.sigmoid(output)
        y_pred_bin = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                             torch.Tensor([1.0]).to(device),
                             torch.Tensor([0.0]).to(device))  
        labels = torch.tensor([label for _, label in data_loader.dataset])
        y = labels[:,3].float()
        t = labels[:,0]
        ead = torch.mean(y[t==1]) - torch.mean(y[t==0])
        ead_prob = torch.mean(y_pred_probs[t==1]) - torch.mean(y_pred_probs[t==0])
        ead_binary = torch.mean(y_pred_bin[t==1]) - torch.mean(y_pred_bin[t==0])
        ATE = 0.3
        TEB = ead_prob - ATE
        TEB_bin = ead_binary - ATE
        acc = accuracy_score(y, y_pred_bin)
        bacc = balanced_accuracy_score(y, y_pred_bin)
        
        if verbose:
            print(f'All set: Accuracy: {acc:.2f}, Balanced Accuracy: {bacc:.2f}, TEB: {TEB:.2f}, TEB_bin: {TEB_bin:.2f}, EAD: {ead:.2f}')
    return acc.item(), bacc.item(), TEB.item(), TEB_bin.item(), ead.item()


def evaluate(model, device, loader, verbose, set_name="test set"):
    '''
    Evaluate the model on the given dataset.

    Args:
        model: torch.nn.Module
        device: str
        loader: torch.utils.data.DataLoader
        verbose: bool
        set_name: str
    '''
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target[3].to(device).float()
            output = model(data)
            loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item() 
            pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                               torch.Tensor([1.0]).to(device),
                               torch.Tensor([0.0]).to(device))  
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader.dataset)
    if verbose:
        print('Performance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            set_name, loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))

    return 100. * correct / len(loader.dataset)

def compute_ead(model, data_loader, device = 'cpu'):
    '''
    Compute Empirical Associational Difference (EAD) for the given 
    model and data_loader.

    Args:
        model: torch.nn.Module
        data_loader: torch.utils.data.DataLoader
        device: str
    '''
    model.eval()
    y = torch.tensor([])
    b = torch.tensor([])
    y_prob = torch.tensor([])
    y_binary = torch.tensor([])
    with torch.no_grad():
        for data, target in data_loader:
            X_b, y_b = data.to(device), target[3].to(device).float()
            b_b = target[0].to(device).float()
            y_prob_b = torch.sigmoid(model(X_b))
            y_binary_b = torch.where(y_prob_b > 0.5, 
                                 torch.tensor(1.0).to(device), 
                                 torch.tensor(0.0).to(device))
            y = torch.cat((y, y_b), 0)
            b = torch.cat((b, b_b), 0)
            y_prob = torch.cat((y_prob, y_prob_b), 0)
            y_binary = torch.cat((y_binary, y_binary_b), 0)
        ead = torch.mean(y[b==1]) - torch.mean(y[b==0])
        ead_prob = torch.mean(y_prob[b==1]) - torch.mean(y_prob[b==0])
        ead_binary = torch.mean(y_binary[b==1]) - torch.mean(y_binary[b==0])
    return ead, ead_prob, ead_binary

def train_epoch(model, device, train_loader, optimizer, epoch, verbose):
    '''
    Train the model for one epoch.

    Args:
        model: torch.nn.Module
        device: str
        train_loader: torch.utils.data.DataLoader
        optimizer: torch.optim.Optimizer
        epoch: int
        verbose: bool
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target[3].to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def training(finetuning=False, force_generation=False, subsampling="random", train_ratio=0.02,
          normalize=False, verbose=True):
    '''
    Train the model on the CausalMNIST dataset.
    
    Args:
        finetuning: bool
        force_generation: bool
        subsampling: str
        normalize: bool

    '''
    use_gpu = torch.cuda.is_available()
    device = torch.device("gpu" if use_gpu else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

    all = CausalMNIST(root='./data', 
                       env='all', 
                       transform=transforms.ToTensor(),
                       force_generation=force_generation,
                       train_ratio=train_ratio,
                       subsampling=subsampling,
                       verbose=verbose)
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if normalize:
        mean = 0
        std = 0
        for img, _ in all:
            mean += img.mean(dim=(1, 2))
            std += img.std(dim=(1, 2))
            mean /= len(all)
            std /= len(all)
        model.transformations = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    else:
        model.transformations = transforms.Compose([transforms.ToTensor()])
    if finetuning:
        train_loader = torch.utils.data.DataLoader(
            CausalMNIST(root='./data', 
                        env='train',
                        transform=model.transformations,
                        subsampling=subsampling,
                        verbose=verbose),
            batch_size=64, shuffle=True, **kwargs)
        
        val_loader = torch.utils.data.DataLoader(
            CausalMNIST(root='./data', 
                        env='val',
                        transform=model.transformations,
                        subsampling=subsampling,
                        verbose=verbose),
            batch_size=1000, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            CausalMNIST(root='./data', 
                        env='train_full',
                        transform=model.transformations,
                        subsampling=subsampling,
                        verbose=verbose),
            batch_size=64, shuffle=True, **kwargs)

    for epoch in range(6):
        train_epoch(model, device, train_loader, optimizer, epoch, verbose)
        evaluate(model, device, train_loader, verbose, set_name='train set')
        if finetuning:
            evaluate(model, device, val_loader, verbose, set_name='val set')
    return model
