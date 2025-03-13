from dataset import CausalMNIST
from models import ConvNet, compute_effect
from train import training
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pandas as pd
import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# conda activate crl
# python src/run.py --k 7 --p 0.7 --exp OS --N 10000 --seeds 5


def get_parser():
    parser = argparse.ArgumentParser(description='Causal MNIST')
    parser.add_argument('--e', type=int, default=1, help='Experiment')
    parser.add_argument('--pW', type=float, default=0.5, help='Probability of W (observed confounders)')
    parser.add_argument('--pU', type=float, default=0.5, help='Probability of U (unobserved confounders)')
    parser.add_argument('--exp', type=str, default='OS', help='Experiment type')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    return parser

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [("ERM", None), ('DERM', None)]#("CURL", 0.1), ("CURL", 10), ("UCRL", 0.1), ("UCRL", 10), ("CURL+", 0.1), ("CURL+", 10)]
    train_ratios = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    results = pd.DataFrame(columns=["exp", "method", "train_ratio", "seed", "tr_acc", "tr_bacc", "val_acc", "val_bacc", "OS_ad_", "OSUC", "OSUC_tr", "OSUC_", "OSOC", "OSOC_tr", "OSOC_", "RCT"])
    i = 0
    t0 = time.time()
    for method, k_inv in methods:
        for train_ratio in train_ratios:
            for seed in range(args.seeds):
                t = time.time()-t0
                N_i = len(methods)*len(train_ratios)*args.seeds
                T = t*N_i/i if i > 0 else 0
                print(f"Training {i+1}/{N_i} {t//60:.0f}m{t%60:.0f}s/{T//60:.0f}m{T%60:.0f}s (Method: {method}, K_inv: {k_inv}, Train Ratio: {train_ratio}, Seed: {seed})")
                dataset = CausalMNIST(root='./data',
                        N=args.N,
                        e=args.e,
                        pW=args.pW,
                        pU=args.pU,
                        exp="RCT",
                        verbose=False,
                        seed=seed,
                        force_generation=False)
                RCT = compute_effect(dataset, method="AIPW", pred=False, total=True, econml=False)
                dataset = CausalMNIST(root='./data',
                                    N=args.N,
                                    e=args.e,
                                    pW=args.pW,
                                    pU=args.pU,
                                    exp=args.exp,
                                    verbose=False,
                                    seed=seed,
                                    force_generation=False)
                model = ConvNet()
                try:
                    model = training(model, 
                                    dataset,
                                    epochs=args.epochs, 
                                    batch_size=64, 
                                    lr=0.0005, 
                                    method=method,
                                    k_inv=k_inv,
                                    verbose=False,
                                    train_ratio=train_ratio,
                                    log_dir=f"./logs/{args.e}/{args.pW}/{args.pU}/{args.exp}/{seed}/{train_ratio}",
                                    eval=False)
                except:
                    print(f"Training failed for {method} (k_inv={k_inv}) with train_ratio={train_ratio}")
                    N_i -=1
                    continue
                A = time.time()
                dataset.Y_hat = model(dataset.X.to(device)).max(axis=1)[1].cpu().numpy()
                B = time.time()
                n_tr = int(train_ratio*len(dataset))
                tr_acc = accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
                tr_bal_acc = balanced_accuracy_score(dataset.Y[:n_tr], dataset.Y_hat[:n_tr])
                val_acc = accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
                val_bal_acc = balanced_accuracy_score(dataset.Y[n_tr:], dataset.Y_hat[n_tr:])
                C = time.time()
                OS_ad_ = compute_effect(dataset, method="AD", pred=True)

                OSOC_ = compute_effect(dataset, method="AIPW", pred=True, total=True)
                OSOC_tr = compute_effect(dataset, method="AIPW", pred=False, total=True, train_ratio=train_ratio)
                OSOC = compute_effect(dataset, method="AIPW", pred=False, total=True)

                OSUC_ = compute_effect(dataset, method="AIPW", pred=True, total=False)
                OSUC_tr = compute_effect(dataset, method="AIPW", pred=False, total=False, train_ratio=train_ratio)
                OSUC = compute_effect(dataset, method="AIPW", pred=False, total=False)
                D = time.time()
                #print(f"Moving Data: {B-A:.3f}s, Stat. Evaluation: {C-B:.3f}s, Causal Evaluation: {D-C:.3f}s")
                print(f'OS UC (pred): {OSUC_:.3f}, OS OC (pred): {OSOC_:.3f}, OS OC train: {OSOC_tr:.3f}, OS OC: {OSOC:.3f}, RCT: {RCT:.3f}')
                results.loc[i] = {"exp": args.exp,
                                "method": method if k_inv is None else f"{method} ({k_inv})",
                                "train_ratio": train_ratio,
                                "seed": seed,
                                "tr_acc": tr_acc,
                                "tr_bacc": tr_bal_acc,
                                "val_acc": val_acc,
                                "val_bacc": val_bal_acc,
                                "OS_ad_": OS_ad_,
                                "OSUC": OSUC,
                                "OSUC_tr": OSUC_tr,
                                "OSUC_": OSUC_,
                                "OSOC": OSOC,
                                "OSOC_tr": OSOC_tr,
                                "OSOC_": OSOC_,
                                "RCT": RCT}
                                
                i += 1
    if not os.path.exists(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}"):
        os.makedirs(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}")
    results.to_csv(f"results/{args.e}/{args.pW}/{args.pU}//{args.exp}/efficiency.csv")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)