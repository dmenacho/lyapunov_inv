import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from pinn_lyap_nn import LyapunovNet, NNDynamics

class StabilityConstrainedInverseSolver:
    def __init__(self, lyapunov, nn_dynamics, device='cpu'):
        self.lyapunov = lyapunov.to(device)
        self.lyapunov.eval()
        self.nn_dynamics = nn_dynamics
        self.device = device
        
        self.num_steps = 15
        self.dt = 1.0 / 0.05 
    
    def simulate_training_trajectory(self, w_init, learning_rate, weight_decay, momentum = 0.0, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        
        model = nn.Sequential(
            nn.Linear(self.nn_dynamics.input_dim, self.nn_dynamics.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.nn_dynamics.hidden_dim, self.nn_dynamics.output_dim)
        ).to(self.device)
        
        self.nn_dynamics.unflatten_params(w_init, model)
        
        trajectory = [torch.tensor(w_init, dtype=torch.float32, device=self.device)]
        
        momentum_buffer = None
        loss_fn = nn.MSELoss()
        
        for step in range(num_steps):
            y_pred = model(self.nn_dynamics.X_train_tensor)
            loss = loss_fn(y_pred, self.nn_dynamics.y_train_tensor)

            loss.backward()

            with torch.no_grad():
                grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.clone())
                    else:
                        grads.append(torch.zeros_like(param))
                
                if momentum > 0:
                    if momentum_buffer is None:
                        momentum_buffer = [torch.zeros_like(g) for g in grads]
                    
                    for i, (param, g) in enumerate(zip(model.parameters(), grads)):
                        momentum_buffer[i] = momentum * momentum_buffer[i] + g
                        param.data -= learning_rate * momentum_buffer[i]
                else:
                    for param, g in zip(model.parameters(), grads):
                        param.data -= learning_rate * g
                
                if weight_decay > 0:
                    for param in model.parameters():
                        param.data -= learning_rate * weight_decay * param.data
                
                for param in model.parameters():
                    param.grad.zero_()
            
            w_current = self.nn_dynamics.flatten_params(model)
            trajectory.append(
                torch.tensor(w_current, dtype=torch.float32, device=self.device)
            )
        
        trajectory = torch.stack(trajectory)
        return trajectory
    
    def compute_loss( self, x0, theta_dict):

        x0_np = x0.detach().cpu().numpy()
        
        # Simulate trajectory
        trajectory = self.simulate_training_trajectory(
            x0_np,
            learning_rate=theta_dict['learning_rate'],
            weight_decay=theta_dict.get('weight_decay', 0.0),
            momentum=theta_dict.get('momentum', 0.0),
            num_steps=self.num_steps
        )
        
        trajectory_normalized = (trajectory - trajectory.mean(dim=0)) / (trajectory.std(dim=0) + 1e-8)
        final_loss_value = trajectory[-1].norm() ** 2
        L_data = final_loss_value
        
        V_values = []
        V_dot_values = []
        
        for i in range(trajectory_normalized.shape[0]):
            state = trajectory_normalized[i:i+1].clone().detach().requires_grad_(True)
            
            V_val = self.lyapunov(state)
            V_values.append(V_val)

            V_grad = torch.autograd.grad(V_val.sum(), state, create_graph=True)[0]
            
            if i < trajectory_normalized.shape[0] - 1:
                f_val = (trajectory_normalized[i+1] - trajectory_normalized[i]) / self.dt
            else:
                f_val = (trajectory_normalized[i] - trajectory_normalized[i-1]) / self.dt
            
            V_dot = (V_grad * f_val).sum()
            V_dot_values.append(V_dot)
        
        V_vals = torch.stack(V_values)
        V_dot_vals = torch.stack(V_dot_values)
        
        x0_normalized = (x0 - x0.mean()) / (x0.std() + 1e-8)
        x0_normalized = x0_normalized.unsqueeze(0)
        roa_constraint_1 = torch.relu(self.lyapunov(x0_normalized))

        alpha = 0.05 
        roa_constraint_2 = torch.relu(V_dot_vals + alpha * V_vals.squeeze()).mean()
        
        L_lyapunov = roa_constraint_1 + roa_constraint_2

        total_loss = L_data + 5.0 * L_lyapunov
        
        return total_loss, L_data, L_lyapunov, final_loss_value
    
    def optimize_single(
        self,
        x0_init: np.ndarray,
        theta_init: Dict[str, float],
        learning_rate_opt: float = 1e-3,
        max_iters: int = 20,
        tol: float = 1e-4
    ) -> Tuple[Dict[str, float], float, float, float]:
        """
        Optimize hyperparameters for a single random initialization.
        
        Args:
            x0_init: Initial weight values
            theta_init: Initial hyperparameter dict
            learning_rate_opt: Optimizer learning rate
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            theta_opt: Optimized hyperparameters dict
            final_loss: Final total loss
            loss_data: Final task loss
            loss_lyapunov: Final Lyapunov loss
        """
        x0 = nn.Parameter(
            torch.tensor(x0_init, dtype=torch.float32, device=self.device)
        )
        
        # Hyperparameters to optimize
        lr_param = nn.Parameter(
            torch.tensor([theta_init['learning_rate']], dtype=torch.float32, device=self.device)
        )
        
        optimizer = optim.Adam([x0, lr_param], lr=learning_rate_opt)
        
        prev_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        for iteration in range(max_iters):
            # Constrain learning rate to valid range
            with torch.no_grad():
                lr_param.clamp_(min=1e-5, max=0.5)
            
            # Create hyperparameter dict
            theta_dict = {
                'learning_rate': lr_param[0].item(),
                'weight_decay': theta_init.get('weight_decay', 0.0),
                'momentum': theta_init.get('momentum', 0.0)
            }
            
            # Compute loss
            total_loss, loss_data, loss_lyapunov, final_loss = self.compute_loss(x0, theta_dict)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([x0, lr_param], max_norm=1.0)
            optimizer.step()
            
            # Early stopping
            if abs(total_loss.item() - prev_loss) < tol:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"  Converged at iteration {iteration+1}")
                    break
            else:
                patience_counter = 0
            
            prev_loss = total_loss.item()
            
            if (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: Loss={total_loss.item():.4f}, "
                      f"L_data={loss_data.item():.4f}, L_lyap={loss_lyapunov.item():.4f}, "
                      f"η={theta_dict['learning_rate']:.4f}")
        
        # Final evaluation
        theta_dict['learning_rate'] = lr_param[0].item()
        total_loss, loss_data, loss_lyapunov, final_loss = self.compute_loss(x0, theta_dict)
        
        return theta_dict, total_loss.item(), loss_data.item(), loss_lyapunov.item()
    
    def discover_admissible_set(
        self,
        n_samples: int = 100,
        learning_rate_range: Tuple[float, float] = (1e-5, 0.5)
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Discover ranges of admissible hyperparameters via inverse optimization.
        
        Args:
            n_samples: Number of random initializations
            learning_rate_range: Range of learning rates to sample
            
        Returns:
            admissible_total: List of admissible hyperparameter dicts (total loss criterion)
            admissible_data: List of admissible hyperparameter dicts (data loss criterion)
        """
        admissible_total = []
        admissible_data = []
        
        # Sample random initial hyperparameters
        lr_samples = np.random.uniform(*learning_rate_range, n_samples)
        
        for i, lr_init in enumerate(tqdm(lr_samples, desc="Inverse Optimization")):
            # Random initial weight
            x0_init = np.random.randn(self.nn_dynamics.state_dim).astype(np.float32) * 0.1
            
            # Initial hyperparameter dict
            theta_init = {'learning_rate': float(lr_init)}
            
            print(f"\n[{i+1}/{n_samples}] Initial η = {lr_init:.4f}")
            
            # Optimize
            theta_opt, loss_total, loss_data, loss_lyapunov = self.optimize_single(
                x0_init,
                theta_init,
                learning_rate_opt=5e-4,
                max_iters=15,
                tol=1e-4
            )
            
            print(f"  → Optimized η = {theta_opt['learning_rate']:.4f}")
            print(f"  → Total Loss = {loss_total:.4f}, Data Loss = {loss_data:.4f}, "
                  f"Lyapunov Loss = {loss_lyapunov:.4f}")
            
            # Accept if losses are low
            if loss_total < 2.0:
                admissible_total.append(theta_opt)
            
            if loss_data < 1.5:
                admissible_data.append(theta_opt)
        
        return admissible_total, admissible_data


def plot_admissible_hyperparameters(
    admissible_configs: List[Dict],
    criterion: str = "total"
):
    """
    Plot distribution of admissible hyperparameters.
    
    Args:
        admissible_configs: List of admissible hyperparameter dicts
        criterion: "total" or "data"
    """
    if not admissible_configs:
        print("No admissible configurations found!")
        return
    
    # Extract learning rates
    learning_rates = np.array([cfg['learning_rate'] for cfg in admissible_configs])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(learning_rates, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].set_xlabel('Learning Rate (η)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of Admissible Learning Rates\n({criterion} criterion)',
                      fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Log-scale histogram
    axes[1].hist(np.log10(learning_rates), bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[1].set_xlabel('log₁₀(η)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Log-scale Distribution\n({criterion} criterion)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'admissible_hyperparameters_{criterion}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


def main():
    """
    Main inverse optimization pipeline.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load trained Lyapunov PINN
    print("[1] Loading trained Lyapunov PINN...")
    nn_dynamics = NNDynamics(
        input_dim=1,
        hidden_dim=8,
        output_dim=1,
        device=device
    )
    
    lyapunov = LyapunovNet(state_dim=nn_dynamics.state_dim, hidden_dim=128).to(device)
    lyapunov.load_state_dict(torch.load('models/lyapunov_nn_training.pt', map_location=device))
    print("✓ Loaded Lyapunov PINN")
    
    # Initialize inverse solver
    print("\n[2] Initializing stability-constrained inverse solver...")
    solver = StabilityConstrainedInverseSolver(lyapunov, nn_dynamics, device=device)
    print("✓ Solver initialized")
    
    # Discover admissible hyperparameters
    print("\n[3] Discovering admissible hyperparameter ranges...")
    admissible_total, admissible_data = solver.discover_admissible_set(
        n_samples=50,
        learning_rate_range=(1e-5, 0.2)
    )
    
    print(f"\n✓ Found {len(admissible_total)} admissible configs (total loss criterion)")
    print(f"✓ Found {len(admissible_data)} admissible configs (data loss criterion)")
    
    # Save results
    print("\n[4] Saving results...")
    Path('results').mkdir(exist_ok=True)
    
    # Convert to arrays for saving
    lr_total = np.array([cfg['learning_rate'] for cfg in admissible_total])
    lr_data = np.array([cfg['learning_rate'] for cfg in admissible_data])
    
    np.save('results/admissible_lr_total.npy', lr_total)
    np.save('results/admissible_lr_data.npy', lr_data)
    print("✓ Saved admissible learning rates")
    
    # Plot results
    print("\n[5] Generating plots...")
    plot_admissible_hyperparameters(admissible_total, criterion="total")
    plot_admissible_hyperparameters(admissible_data, criterion="data")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if len(lr_total) > 0:
        print(f"\nTotal Loss Criterion (n={len(lr_total)}):")
        print(f"  Learning Rate Range: [{lr_total.min():.4e}, {lr_total.max():.4e}]")
        print(f"  Mean:  {lr_total.mean():.4e}")
        print(f"  Std:   {lr_total.std():.4e}")
        print(f"  Median: {np.median(lr_total):.4e}")
    
    if len(lr_data) > 0:
        print(f"\nData Loss Criterion (n={len(lr_data)}):")
        print(f"  Learning Rate Range: [{lr_data.min():.4e}, {lr_data.max():.4e}]")
        print(f"  Mean:  {lr_data.mean():.4e}")
        print(f"  Std:   {lr_data.std():.4e}")
        print(f"  Median: {np.median(lr_data):.4e}")
    
    print("\n✓ Inverse optimization complete!")


if __name__ == "__main__":
    main()