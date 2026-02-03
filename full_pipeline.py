import argparse
import csv
import json
import os

import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import argparse
import random


from src.dynamic_models import TinyResNet
from src import dynamic_models
from src.lyap_models import LyapunovNet
from src.learners import ModelOCIFAR10Dynamics,LyapunovLearner
from src import learners
from src.inverser_modeling import StabilityConstrainedNN
from src.vis_utils import plot_stable_weights


def set_model_weights(model, w_vec,device):
    """w_vec: numpy [state_dim] from your admissible set"""
    w_vec = np.asarray(w_vec).ravel()
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = w_vec[idx:idx+n]
            p.copy_(torch.tensor(chunk.reshape(p.shape), dtype=torch.float32, device=device))
            idx += n

def train_one_run(model, train_loader, test_loader, n_epochs=10, lr=0.01, device = 'cpu'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_accs = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        train_losses.append(running_loss / n_batches)
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        test_accs.append(acc)
    return np.array(train_losses), np.array(test_accs)


def build_parser():
    p = argparse.ArgumentParser("lyap_resnet_pipeline")

    # global
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./training_info")

    sub = p.add_subparsers(dest="command", required=True)

    # ONE command: full pipeline
    f = sub.add_parser("full", help="Run full pipeline: train Lyapunov then stable-search.")

    f.add_argument("--run_id", type=str, default="")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    f.add_argument("--data_dir", type=str, default="./data")
    f.add_argument("--out_dir", type=str, default="./training_info")

    # --- train phase args ---
    f.add_argument("--dataset", type=str, default="CIFAR10")
    f.add_argument("--target_model_name", type=str, default="TinyResNet")
    f.add_argument("--num_train_samples", type=int, default=10000)
    f.add_argument("--num_trajectories", type=int, default=200)
    f.add_argument("--traj_steps", type=int, default=50)
    f.add_argument("--gd_lr", type=float, default=1e-2)
    #f.add_argument("--data_type", type=str, default="CIFAR10", choices=["cifar", "nmist"])

    f.add_argument("--w_init_scale", type=float, default=0.5)

    f.add_argument("--lyap_lr", type=float, default=1e-3)
    f.add_argument("--lyap_hidden_dim", type=int, default=256)
    f.add_argument("--lyap_epochs", type=int, default=100)
    f.add_argument("--lyap_batch_size", type=int, default=128)
    f.add_argument("--zubov_loss_mu", type=float, default=0.1)
    f.add_argument("--zubov_loss_transform", type=str, default="exp", choices=["exp", "poly"])
    f.add_argument("--grad_clip", type=float, default=1.0)

    # --- stable search phase args ---
    #f.add_argument("--num_train_samples_search", type=int, default=10000, help="CIFAR samples used during stable-search dynamics.")
    f.add_argument("--search_samples", type=int, default=1000)
    f.add_argument("--weight_scale_min", type=float, default=0.1)
    f.add_argument("--weight_scale_max", type=float, default=1.0)
    f.add_argument("--alpha", type=float, default=0.05)
    f.add_argument("--constraint_threshold", type=float, default=1.0)
    f.add_argument("--verbose_every", type=int, default=5)

    f.add_argument("--test_comparative_epochs", type=int, default=50, help="number of epochs to test method.")
    f.add_argument("--test_comparative_lr", type=float, default=0.01)

    return p

def main():

    parser = build_parser()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # run id
    if not args.run_id:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")

    # save args
    run_dir_params = Path(args.out_dir) / "params" / args.run_id
    run_dir_params.mkdir(parents=True, exist_ok=True)
    (run_dir_params / "params.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################################################################################################
    # DYNAMIC SAMPLING
    ######################################################################################################
    
    #nn_dynamics = ModelOcifar10Dynamics(TinyResNet().to(device), device=device, num_train_samples=1000)
    #nn_dynamics = ModelOcifar10Dynamics(TinyResNet().to(device), device=device, num_train_samples=1000)
    try:
        #nn_dynamics = getattr(learners,f"ModelO{args.dataset}10Dynamics")(TinyResNet().to(device), device=device, num_train_samples=1000)
        nn_dynamics = getattr(learners,f"ModelO{args.dataset}10Dynamics")(getattr(dynamic_models,args.target_model_name)().to(device), device=device, num_train_samples=1000)
    except:
        print(f"ModelO{args.dataset}Dynamics class has been not implemented")
    print(f" State dimension: {nn_dynamics.state_dim}")

    x_train, f_train, aux = nn_dynamics.generate_collocation_points(num_trajectories=args.num_trajectories, num_steps=args.traj_steps)

    loss=aux[0]

    plt.figure(figsize=(10, 6))
    plt.plot(loss, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Dynamic model (target model) loss)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(f'{args.data_dir}/original_dynamic_traning_sample_{args.target_model_name}.png', dpi=150, bbox_inches='tight')

    ######################################################################################################
    # LYAPONUV LEARNING
    ######################################################################################################
    
    lyap_model = LyapunovNet(state_dim= nn_dynamics.state_dim , hidden_dim=256).to(device)
    optimizer = optim.Adam(lyap_model.parameters(), lr=args.lyap_lr)
    #learner = LyapunovLearner(state_dim=nn_dynamics.state_dim, device=device)
    learner = LyapunovLearner(lyap_model, optimizer , state_dim = nn_dynamics.state_dim, device= device)
    losses = learner.train(
        x_train,
        f_train,
        epochs=args.lyap_epochs,
        batch_size=args.lyap_batch_size,
        mu = args.zubov_loss_mu,
        transform=args.zubov_loss_transform,
        grad_clip=args.grad_clip
        )
    torch.save(learner.lyapunov.state_dict(), f'{args.data_dir}/lyapunov_{args.target_model_name}.pt')

    run_dir = Path(args.out_dir) / "models" / f"{args.num_trajectories}_{args.traj_steps}_{args.gd_lr}" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "lyapunov_checkpoint.pt"
    torch.save(
        {
            "lyapunov_state_dict": learner.lyapunov.state_dict(),
            "stats": {k: torch.tensor(v, dtype=torch.float32) for k, v in stats.items()},
            "args": vars(args),
            "losses": losses,
        },
        ckpt_path,
    )
    ckpt_path_model = run_dir / "lyapunov_{args.target_model_name}.pt"
    torch.save(learner.lyapunov.state_dict(), ckpt_path_model)

    ######################################################################################################
    # INVERSE PROBLEM
    ######################################################################################################

    device = torch.device(args.device)

    print(f"[stable] state_dim={nn_dynamics.state_dim}")

    obj = torch.load(ckpt_path, map_location=device)
    lyap = LyapunovNet(state_dim=nn_dynamics.state_dim, hidden_dim=args.lyap_hidden_dim).to(device)
    lyap.load_state_dict(obj["lyapunov_state_dict"])
    stats = obj.get("stats", {})

    solver = StabilityConstrainedNN(
        lyapunov=lyap,
        resnet_dynamics=nn_dynamics,
        learning_rate=args.gd_lr,
        traj_steps=args.traj_steps,
        device=device,
        alpha=args.alpha,
        constraint_threshold=args.constraint_threshold,
    )
    if "x_mean" in stats and "x_std" in stats:
        solver.set_normalization_stats(stats["x_mean"], stats["x_std"], stats.get("f_mean", None), stats.get("f_std", None))
    else:
        print("[stable] warning: no x_mean/x_std in checkpoint; using fallback normalization.")

    stable_weights, stable_meta = solver.discover_stable_initializations(
        n_samples=args.search_samples,
        weight_scale_range=(args.weight_scale_min, args.weight_scale_max),
        verbose_every=args.verbose_every,
    )

    # Save artifacts
    results_dir = Path(args.out_dir) /  "results" / f"{args.num_trajectories}_{args.traj_steps}_{args.gd_lr}" /args.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(stable_weights) > 0:
        np.save(results_dir / "stable_weights.npy", np.asarray(stable_weights, dtype=np.float32))
        (results_dir / "stable_weights_metadata.json").write_text(
            json.dumps({"items": stable_meta}, indent=2)
        )
        plot_stable_weights(stable_meta, results_dir / "stable_weights_plots.png")
        print(f"[stable] saved stable results to: {results_dir}")
    else:
        print("[stable] no stable weights found")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Lyapunov PINN Training Loss (ResNet Training Dynamics)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(f'{args.data_dir}/lyapunov_resnet_training_loss.png', dpi=150, bbox_inches='tight')

    ######################################################################################################
    # COMPARATIVE TRAINING
    ######################################################################################################

    stable_weights = np.load(results_dir / "stable_weights.npy")  
    print("Num stable weights:", len(stable_weights))
    print("State dim:", stable_weights[0].shape[0])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    #train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = getattr(datasets,args.dataset)(root='./data', train=True, download=True, transform=transform)
    #test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_dataset = getattr(datasets,args.dataset)(root='./data', train=False, download=True, transform=transform)

    n_train = 10000
    n_test  = 500
    train_idx = np.random.choice(len(train_dataset), n_train, replace=False)
    test_idx  = np.random.choice(len(test_dataset),  n_test,  replace=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=64, shuffle=True)
    test_loader  = DataLoader(Subset(test_dataset,  test_idx),  batch_size=64, shuffle=False)


    n_epochs = args.test_comparative_epochs
    lr = args.test_comparative_lr

    n_runs = min(10, len(stable_weights))   # e.g. test first 50 admissible weights

    losses_stable = []
    accs_stable = []

    for i in tqdm(range(n_runs), desc="Stable inits"):
        w = stable_weights[i]
        model = getattr(dynamic_models,args.target_model_name)().to(device)()
        set_model_weights(model, w, device)
        tr_loss, te_acc = train_one_run(model, train_loader, test_loader, n_epochs=n_epochs, lr=lr, device= device)
        losses_stable.append(tr_loss)
        accs_stable.append(te_acc)

    losses_stable = np.stack(losses_stable)
    accs_stable   = np.stack(accs_stable)

    losses_random = []
    accs_random = []

    for i in tqdm(range(n_runs), desc="Random inits"):
        model = getattr(dynamic_models,args.target_model_name)().to(device)()
        tr_loss, te_acc = train_one_run(model, train_loader, test_loader, n_epochs=n_epochs, lr=lr, device= device)
        losses_random.append(tr_loss)
        accs_random.append(te_acc)

    losses_random = np.stack(losses_random)
    accs_random   = np.stack(accs_random)

    epochs = np.arange(1, n_epochs+1)

    plt.figure(figsize=(12,5))

    # Train loss
    plt.subplot(1,2,1)
    m_stable = losses_stable.mean(axis=0)
    s_stable = losses_stable.std(axis=0)
    m_rand   = losses_random.mean(axis=0)
    s_rand   = losses_random.std(axis=0)

    plt.plot(epochs, m_stable, label="Lyapunov init")
    plt.fill_between(epochs, m_stable - s_stable, m_stable + s_stable, alpha=0.2)

    plt.plot(epochs, m_rand, label="Random init")
    plt.fill_between(epochs, m_rand - s_rand, m_rand + s_rand, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.yscale("log")
    plt.title(f"Training convergence ({args.dataset}, {args.target_model_name})")
    plt.grid(alpha=0.3)
    plt.legend()

    # Test accuracy
    plt.subplot(1,2,2)
    m_stable_acc = accs_stable.mean(axis=0)
    s_stable_acc = accs_stable.std(axis=0)
    m_rand_acc   = accs_random.mean(axis=0)
    s_rand_acc   = accs_random.std(axis=0)

    plt.plot(epochs, m_stable_acc, label="Lyapunov init")
    plt.fill_between(epochs, m_stable_acc - s_stable_acc, m_stable_acc + s_stable_acc, alpha=0.2)

    plt.plot(epochs, m_rand_acc, label="Random init")
    plt.fill_between(epochs, m_rand_acc - s_rand_acc, m_rand_acc + s_rand_acc, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Test accuracy over epochs")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{args.data_dir}/Comparative_training.png', dpi=150, bbox_inches='tight')

    # Final accuracy histograms
    final_acc_stable = accs_stable[:, -1]
    final_acc_random = accs_random[:, -1]

    plt.figure(figsize=(6,4))
    bins = np.linspace(0, 100, 21)
    plt.hist(final_acc_stable, bins=bins, alpha=0.6, label="Lyapunov init", edgecolor="black")
    plt.hist(final_acc_random, bins=bins, alpha=0.6, label="Random init", edgecolor="black")
    plt.xlabel("Final test accuracy (%)")
    plt.ylabel("Frequency")
    plt.title("Final performance distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f'{args.data_dir}/Final performance distribution.png', dpi=150, bbox_inches='tight')

    print("Lyapunov init: mean final acc = ", final_acc_stable.mean())
    print("Random init:   mean final acc = ", final_acc_random.mean())

    


if __name__ == "__main__":
    main()