from dataset import CausalMNIST
from models import ConvNet, compute_effect
from train import training
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pandas as pd
import os
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# conda activate crl
# python src/run.py --k 7 --p 0.7 --exp OS --N 10000 --seeds 5

def get_parser():
    parser = argparse.ArgumentParser(description='Causal MNIST')
    parser.add_argument('--clip', type=float, default=0.005, help='Clip propensity')
    parser.add_argument('--e', type=int, default=1, help='Experiment')
    parser.add_argument('--pW', type=float, default=0.5, help='Probability of W (observed confounders)')
    parser.add_argument('--pU', type=float, default=0.5, help='Probability of U (unobserved confounders)')
    parser.add_argument('--exp', type=str, default='OS', help='Experiment type')
    parser.add_argument('--N', type=int, default=10000, help='Number of samples')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [("DERM", None), ('ERM', None), ("UCRL", 0.1)]#("CURL", 0.1), ("CURL", 10), ("UCRL", 0.1), ("UCRL", 10), ("CURL+", 0.1), ("CURL+", 10)]
    results = pd.DataFrame(columns=["target", "method", "seed", "acc", "bacc", "AD", "PPAIPW_OC", "AIPW_OC", "PPAIPW_UC", "AIPW_UC", "ATE"])
    i = 0
    target_names = ["RT","RV","T1","T2","T3"]
    t0 = time.time()
    N_i = len(methods)*args.seeds

    colors = ['lightgray', 'red', 'yellow', 'green']  # Modify colors as desired
    custom_cmap = LinearSegmentedColormap.from_list('custom_gray_to_color', colors)
    for method, k_inv in methods:
        for seed in range(args.seeds):
            t = time.time()-t0
            T = t*N_i/(i/len(target_names)) if i > 0 else 0
            print(f"Training {int(i/len(target_names)+1)}/{N_i} {t//60:.0f}m{t%60:.0f}s/{T//60:.0f}m{T%60:.0f}s (Method: {method}, K_inv: {k_inv}, Seed: {seed})")
            reference = CausalMNIST(root='./data',
                                N=args.N,
                                e=args.e,
                                pW=args.pW,
                                pU=args.pU,
                                exp=args.exp,
                                verbose=False,
                                seed=seed,
                                force_generation=False,
                                clip=args.clip)
            model = ConvNet()
            try:
                model = training(model, 
                                reference,
                                epochs=args.epochs, 
                                batch_size=32, 
                                lr=0.0001, 
                                method=method,
                                k_inv=k_inv,
                                verbose=False,
                                train_ratio=1,
                                log_dir=f"./logs/{args.e}/{args.pW}/{args.pU}/{args.exp}/{seed}",
                                eval=False)
            except:
                print(f"Training failed for {method} (k_inv={k_inv})")
                N_i -= len(target_names)
                continue

            for target_name in target_names:
                if target_name=="RT":
                    target_RCT = CausalMNIST(root='./data',
                            N=args.N,
                            e=args.e,
                            pW=args.pW,
                            pU=args.pU,
                            exp=args.exp,
                            verbose=False,
                            seed=seed,
                            force_generation=False)
                    ATE = compute_effect(target_RCT, method="AIPW", pred=False, total=True, econml=False)
                    target = reference
                elif target_name=="RV":
                    target = CausalMNIST(root='./data',
                            N=args.N,
                            e=args.e,
                            pW=args.pW,
                            pU=args.pU,
                            exp=args.exp,
                            verbose=False,
                            seed=seed+1,
                            force_generation=False)
                elif target_name=="T3": # difficult (strong OoD, OS)
                    target_RCT = CausalMNIST(root='./data',
                                N=args.N,
                                e=2 if args.e==1 else 1,
                                pW=0.5,
                                pU=0.5,
                                exp="RCT",
                                verbose=False,
                                seed=seed,
                                force_generation=False)
                    ATE = compute_effect(target_RCT, method="AIPW", pred=False, total=True, econml=False)
                    target = CausalMNIST(root='./data',
                                N=args.N,
                                e=2 if args.e==1 else 1,
                                pW=0.5,
                                pU=0.5,
                                exp="OS",
                                verbose=False,
                                seed=seed,
                                force_generation=False)
                elif target_name=="T2": # medium (weak OoD, OS)
                    target_RCT = CausalMNIST(root='./data',
                                N=args.N,
                                e=2 if args.e==1 else 1,
                                pW=0.05,
                                pU=0.05,
                                exp="RCT",
                                verbose=False,
                                seed=seed,
                                force_generation=False)
                    ATE = compute_effect(target_RCT, method="AIPW", pred=False, total=True, econml=False)
                    target = CausalMNIST(root='./data',
                                N=args.N,
                                e=2 if args.e==1 else 1,
                                pW=0.05,
                                pU=0.05,
                                exp="OS",
                                verbose=False,
                                seed=seed,
                                force_generation=False)
                elif target_name=="T1": # easy (weak OoD, RCT)
                    target_RCT = CausalMNIST(root='./data',
                                N=args.N,
                                e=2 if args.e==1 else 1,
                                pW=0.05,
                                pU=0.05,
                                exp="RCT",
                                verbose=False,
                                seed=seed,
                                force_generation=False)
                    ATE = compute_effect(target_RCT, method="AIPW", pred=False, total=True, econml=False)
                    target = target_RCT
                target.Y_hat = model(target.X.to(device)).max(axis=1)[1].cpu().numpy()
                Z = target.W*1+target.T*2
                Z_ref = reference.W*1+reference.T*2
                fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                axs[0].imshow(np.array([[((reference.Y==j)*(Z_ref==k)).sum().item() for j in range(10)] for k in range(4)]), cmap=custom_cmap)
                for k in range(4):
                    for j in range(10):
                        axs[0].text(j, k, str(round(((reference.Y==j)*(Z_ref==k)).sum().item(),2)), fontsize=12, color='w', ha='center', va='center')
                axs[0].set_xlabel('Y')
                axs[0].set_yticks(range(4))
                axs[0].set_yticklabels(['T=0, W=0', 'T=0, W=1', 'T=1, W=0', 'T=1, W=1'])
                axs[0].set_xticks(range(10))
                axs[0].set_xticklabels(range(10))
                axs[0].set_title(r'Reference $\mathbb{P}(Z,Y) \cdot N$')

                axs[1].imshow(np.array([[((target.Y==j)*(Z==k)).sum().item() for j in range(10)] for k in range(4)]), cmap=custom_cmap)
                for k in range(4):
                    for j in range(10):
                        axs[1].text(j, k, str(round(((target.Y==j)*(Z==k)).sum().item(),2)), fontsize=12, color='w', ha='center', va='center')
                axs[1].set_xlabel('Y')
                axs[1].set_yticks(range(4))
                axs[1].set_yticklabels(['T=0, W=0', 'T=0, W=1', 'T=1, W=0', 'T=1, W=1'])
                axs[1].set_xticks(range(10))
                axs[1].set_xticklabels(range(10))
                axs[1].set_title(r'Target $\mathbb{P}(Z,Y) \cdot N$')
                axs[2].imshow(np.array([[((target.Y==j)*(Z==k)*(target.Y==target.Y_hat)).sum().item()/((target.Y==j)*(Z==k)).sum().item() if ((target.Y==j)*(Z==k)).sum().item() > 0 else 0 for j in range(10)] for k in range(4)]), cmap=custom_cmap)
                for k in range(4):
                    for j in range(10):
                        N_k_j = ((target.Y==j)*(Z==k)).sum().item()
                        acc_k_j = ((target.Y==j)*(Z==k)*(target.Y==target.Y_hat)).sum().item()/N_k_j if N_k_j > 0 else 0
                        axs[2].text(j, k, str(round(acc_k_j,2)), fontsize=12, color='w', ha='center', va='center')
                axs[2].set_xlabel('Y')
                axs[2].set_yticks(range(4))
                axs[2].set_yticklabels(['T=0, W=0', 'T=0, W=1', 'T=1, W=0', 'T=1, W=1'])
                axs[2].set_xticks(range(10))
                axs[2].set_xticklabels(range(10))
                axs[2].set_title(r'Target $\mathbb{P}(g(X)=Y|Z,Y)$')
                if not os.path.exists(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}"):
                    os.makedirs(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}")
                plt.subplots_adjust(hspace=0.3) 
                plt.savefig(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}/{target_name}_{method}_{seed}.png", bbox_inches='tight')
                plt.close()

                acc = accuracy_score(target.Y, target.Y_hat)
                bacc = balanced_accuracy_score(target.Y, target.Y_hat)

                AD = compute_effect(target, method="AD", pred=False, total=None)
                
                PPAIPW_OC = compute_effect(target, method="AIPW", pred=True, total=True)
                AIPW_OC = compute_effect(target, method="AIPW", pred=False, total=True)

                PPAIPW_UC = compute_effect(target, method="AIPW", pred=True, total=False)
                AIPW_UC = compute_effect(target, method="AIPW", pred=False, total=False)
                print(f'{target_name} - Acc: {acc:.3f}, PPAIPW (OC): {PPAIPW_OC:.3f}, PPAIPW (UC): {PPAIPW_UC:.3f}, AIPW (OC): {AIPW_OC:.3f}, AIPW (UC): {AIPW_UC:.3f}, ATE (RCT): {ATE:.3f}')
                results.loc[i] = {"target": target_name,
                                  "method": method if k_inv is None else f"{method} ({k_inv})",
                                  "seed": seed,
                                  "acc": acc,
                                  "bacc": bacc,
                                  "AD": AD,
                                  "PPAIPW_OC": PPAIPW_OC,
                                  "AIPW_OC": AIPW_OC,
                                  "PPAIPW_UC": PPAIPW_UC,
                                  "AIPW_UC": AIPW_UC,
                                  "ATE": ATE}
                                    
                i += 1
    if not os.path.exists(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}"):
        os.makedirs(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}")
    results.to_csv(f"results/{args.e}/{args.pW}/{args.pU}/{args.exp}/generalization.csv")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)