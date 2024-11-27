import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from models import compute_effect

def training(model,
             dataset, 
             train_ratio=0.9,
             epochs=6,
             lr=0.001,
             batch_size=64,
             method='ERM',
             verbose=True):
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
    device = torch.device("gpu" if use_gpu else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n_tr = int(train_ratio*len(dataset))
    train = dataset.data_label_tuples[:n_tr]
    #val = dataset.data_label_tuples[n_tr:]
    train_loader = torch.utils.data.DataLoader(train, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               **kwargs)
    # val_loader = torch.utils.data.DataLoader(val,
    #                                          batch_size=1000, 
    #                                          shuffle=True, 
    #                                          **kwargs)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (image, variables) in enumerate(train_loader):
            X, y = image.to(device).float(), variables[3].to(device).long()
            optimizer.zero_grad()
            output = model(X)
            loss = torch.nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
        
        # measure accuracy
        model.eval()
        with torch.no_grad():
            dataset.Y_hat = model(dataset.X).max(axis=1)[1].numpy()
            tr_acc = accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
            tr_bal_acc = balanced_accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
            val_acc = accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
            val_bal_acc = balanced_accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
            print(f'Train accuracy: {tr_acc:.2f}, Train balanced accuracy: {tr_bal_acc:.2f}')
            print(f'Validation accuracy: {val_acc:.2f}, Validation balanced accuracy: {val_bal_acc:.2f}')
            AD_ = compute_effect(dataset, method="AD", pred=True)
            AF_ = compute_effect(dataset, method="AF", pred=True)
            print(f'Pred. AD: {AD_:.2f}, Pred. AF: {AF_:.2f}')

    # TODO: return best model
    return model   



