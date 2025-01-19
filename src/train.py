import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from models import compute_effect
from itertools import compress
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd

def training(model,
             dataset, 
             train_ratio=0.9,
             epochs=6,
             lr=0.001,
             batch_size=64,
             k_inv=0.1,
             method='ERM',
             verbose=True,
             log_dir='runs',
             eval=True,
             gpu=True):
    # TODO: update description
    '''
    Train the model on the CausalMNIST dataset.
    
    Args:
        finetuning: bool
        force_generation: bool
        subsampling: str
        normalize: bool

    '''
    use_gpu = torch.cuda.is_available()
    if gpu:
        device = torch.device("cuda" if use_gpu else "cpu")
    else:
        device = torch.device("cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_gpu else {}
    model = model.to(device)
    model.device = device
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr)
    n_tr = int(train_ratio*len(dataset))
    A = time.time()
    train = dataset.data_label_tuples[:n_tr]
    train_loader = torch.utils.data.DataLoader(train, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        **kwargs)
    B = time.time()
    if method in ["CURL", "UCRL"]:
        train_envs = []
        for w in dataset.W.unique():
            for u in dataset.U.unique():
                for t in dataset.T.unique():
                    mask = (dataset.W == w) & (dataset.U == u) & (dataset.T == t)
                    if sum(mask) < batch_size:
                        continue
                    train_env = (dataset.X[mask], dataset.Y[mask])
                    train_envs.append(train_env)
        if verbose: print("Num Env.", len(train_envs))
    # if method in ["CURL+"]:
    #     train_envs = []
    #     for t in dataset.T.unique():
    #         train_env = (dataset.X[dataset.T == t], dataset.Y[dataset.T == t])
    #         train_envs.append(train_env)
    C = time.time()
    if verbose: print(f"Data loading: {B-A:.2f}s")
    if verbose: print(f"Data loading Multi-Env: {C-B:.2f}s")
    writer = SummaryWriter(log_dir=f"{log_dir}/{method if 'ERM' in method else f'{method}_{k_inv}'}")  # Specify log directory
    for epoch in range(epochs):
        D = time.time()
        model.train()
        for batch_idx, (image, variables) in enumerate(train_loader):
            I = time.time()
            X, y = image.to(device).float(), variables[3].to(device).long()
            optimizer.zero_grad()
            output = model(X)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            if method=="DERM":
                loss = torch.nn.CrossEntropyLoss(reduction='none')(output, y)
                yvar = variables[4].to(device).float()
                oprob = variables[5].to(device).float()
                weight = yvar/oprob#+0.0001
                loss = (weight*loss).sum()
            else:
                loss = torch.nn.CrossEntropyLoss()(output, y)
            F = time.time()
            if method in ["CURL", "UCRL"]:
                losses = []
                for train_env in train_envs:
                    idx = torch.randperm(len(train_env[0]))[:batch_size]
                    X, y =  train_env[0][idx].to(device).float(), train_env[1][idx].to(device).float()
                    if method=="CURL":
                        output = model.cond_exp(X)
                        loss_e = (y-output).mean()
                    if method=="UCRL":
                        output = model(X)
                        loss_e = torch.nn.CrossEntropyLoss()(output, y.long())
                    losses.append(loss_e)
                loss_var = torch.var(torch.stack(losses))
                loss += k_inv * loss_var
            G = time.time()
            loss.backward()
            optimizer.step()
            H = time.time()
            if batch_idx % 100 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        E = time.time()
        if verbose: print(f"Epoch {epoch} Train: {E-D:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Multi-Env Inference: {G-F:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Single Inference: {F-I:.2f}s")
        if verbose: print(f"Epoch {epoch} Train Multi-Env Backward: {H-G:.2f}s")
        if eval:
            model.eval()
            with torch.no_grad():
                dataset.Y_hat = model(dataset.X.to(device)).max(axis=1)[1].cpu().numpy()
                tr_acc = accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
                tr_bal_acc = balanced_accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
                val_acc = accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
                val_bal_acc = balanced_accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
                if verbose: print(f'Train Accuracy: {tr_acc:.3f}, Train Bal. Accuracy: {tr_bal_acc:.3f}')
                if verbose: print(f'Val Accuracy: {val_acc:.3f}, Val Bal. Accuracy: {val_bal_acc:.3f}')
                AD_ = compute_effect(dataset, method="AD", pred=True)
                AIPW_ = compute_effect(dataset, method="AIPW", pred=True, total=True)
                ATE = compute_effect(dataset, method="AIPW", pred=False, total=True)
                if verbose: print(f'AD (pred): {AD_:.3f}, AIPW (pred): {AIPW_:.3f}, AIPW: {ATE:.3f}')
                writer.add_scalar('Train - Accuracy', tr_acc, epoch)
                writer.add_scalar('Train - Bal. Accuracy', tr_bal_acc, epoch)
                writer.add_scalar('Val - Accuracy', val_acc, epoch)
                writer.add_scalar('Val - Bal. Accuracy', val_bal_acc, epoch)
                writer.add_scalar('AD', AD_, epoch)
                writer.add_scalar('AIPW', AIPW_, epoch)
                writer.add_scalar('ATE', ATE, epoch)

    writer.close()
    # TODO: return best model
    # TODO: save best model
    return model   



