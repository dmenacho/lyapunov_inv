import numpy as np
from scipy.integrate import solve_ivp, odeint
import pickle
from pathlib import Path
from tqdm import tqdm
import json

class DoubleSpringSystem:
    def __init__(self):
        self.Tx = 0.0
        self.Ty = 2.0
        self.R1 = 1.0 
        self.R2 = 1.0
        self.g = 9.8
    
    def dynamics(self, t, state, theta):
        
        u1x, u1y, u2x, u2y, v1x, v1y, v2x, v2y = state
        m1, m2, k1, k2, b1, b2 = theta
        
        # SPRING 1
        dx1 = u1x - self.Tx
        dy1 = u1y - self.Ty
        L1 = np.sqrt(dx1**2 + dy1**2)
        L1 = max(L1, 1e-8)
        difference_S1 = L1 - self.R1
        ux1 = dx1 / L1
        uy1 = dy1 / L1
        F_spring1_x = -k1 * difference_S1 * ux1
        F_spring1_y = -k1 * difference_S1 * uy1
        
        # SPRING 2
        dx2 = u2x - u1x
        dy2 = u2y - u1y
        L2 = np.sqrt(dx2**2 + dy2**2)
        L2 = max(L2, 1e-8)
        difference_S2 = L2 - self.R2
        ux2 = dx2/L2
        uy2 = dy2/L2
        F_spring2_x_on_1= k2 * difference_S2 * ux2
        F_spring2_y_on_1= k2 * difference_S2 * uy2
        F_spring2_x= -F_spring2_x_on_1
        F_spring2_y= -F_spring2_y_on_1
        
        #DAMPING FORCES
        damping1_x=-b1*v1x
        damping1_y=-b1*v1y
        damping2_x=-b2*v2x
        damping2_y=-b2*v2y
        
        #GRAVITY FORCES
        g_1y = -m1*self.g
        g_2y = -m2*self.g
        
        #SUM FORCES
        Total_F1x = F_spring1_x + F_spring2_x_on_1 + damping1_x
        Total_F1y = F_spring1_y + F_spring2_y_on_1 + damping1_y + g_1y
        a1x = Total_F1x / m1
        a1y = Total_F1y / m1
        Total_F2x = F_spring2_x + damping2_x
        Total_F2y = F_spring2_y + damping2_y + g_2y
        a2x = Total_F2x / m2
        a2y = Total_F2y / m2
        return [v1x, v1y, v2x, v2y, a1x, a1y, a2x, a2y]
        
        
    def equilibrium(self, theta):
        m1, m2, k1, k2, _, _ = theta
        u1y_eq = self.Ty - self.R1 - (m1 + m2) * self.g / k1
        u2y_eq = u1y_eq - self.R2 - m2 * self.g / k2
        ux1_eq = self.Tx
        ux2_eq = self.Tx
        return [ux1_eq, u1y_eq, ux2_eq, u2y_eq, 0.0, 0.0, 0.0, 0.0]
    
    def energy(self, state, theta):
        u1x, u1y, u2x, u2y, v1x, v1y, v2x, v2y = state
        m1, m2, k1, k2, _, _ = theta
        KE = 0.5 * m1 * (v1x**2 + v1y**2) + 0.5 * m2 * (v2x**2 + v2y**2)

        PE_grav = m1 * self.g * u1y + m2 * self.g * u2y
        L1 = np.sqrt((u1x-self.Tx)**2 + (u1y-self.Ty)**2)
        L2 = np.sqrt((u2x-u1x)**2 + (u2y-u1y)**2)
        difference_S1 = L1-self.R1
        difference_S2 = L2-self.R2
        PE_spring = 0.5 * k1*difference_S1**2 + 0.5 * k2 * difference_S2**2
        PE = PE_grav + PE_spring
        return KE + PE
    
class DataGenerator:
    def __init__(self, sim_time=15.0, dt=0.01):
        self.system = DoubleSpringSystem()
        self.sim_time = sim_time
        self.dt = dt
        self.t_eval = np.arange(0, sim_time, dt)
        self.n_timesteps = len(self.t_eval)

    def sample_parameters(self):
        return np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(0.3, 0.7),
            np.random.uniform(4.0, 8.0),
            np.random.uniform(4.0, 8.0),
            np.random.uniform(-0.2, 0.5),
            np.random.uniform(-0.2, 0.5),
        ])
        
    def sample_initial_condition(self, theta, scale=0.1):
        
        x_eq = self.system.equilibrium(theta)
        perturbation = np.zeros(8)
        perturbation[:4] = np.random.randn(4)*scale
        perturbation[4:] = np.random.randn(4)*scale*0.5 
        return x_eq + perturbation
    
    def simulate(self, theta, x0, verbose=False):
        try:
            sol = solve_ivp(fun=lambda t, y: self.system.dynamics(t, y, theta), t_span=(0, self.sim_time),
                            y0=x0, t_eval=self.t_eval, method='RK45',rtol=1e-7, atol=1e-10, dense_output=False)
            trajectory = sol.y.T 
            return trajectory.astype(np.float32), True
            
        except Exception as e:
            if verbose:
                print(f"Simulation failed: {e}")
            return None, False
    
    def check_stability(self, trajectory, theta, pos_tol=0.5, vel_tol=0.2, energy_ratio=0.8):
        x_eq = self.system.equilibrium(theta)
        n_final = max(1, len(trajectory)//5)
        final_states = trajectory[-n_final:]
        
        pos_error = np.mean([np.linalg.norm(state[:4] - x_eq[:4]) for state in final_states])
        vel_mag = np.mean([np.linalg.norm(state[4:]) for state in final_states])
        
        E_initial = self.system.energy(trajectory[0], theta)
        E_final = self.system.energy(trajectory[-1], theta)
      
        energy_stable = (E_final < E_initial*energy_ratio) or (E_initial < 0.01)
        is_stable = (pos_error < pos_tol and vel_mag < vel_tol and energy_stable)
        
        return is_stable
    
    def generate_dataset(self, n_trajectories=10000, output_dir='data', output_name='training_data'):

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        dataset = []
        stable_count = 0
        unstable_count = 0
        failed_count = 0
        pbar = tqdm(total=n_trajectories, desc="Generating trajectories")
        while len(dataset) < n_trajectories:
            theta = self.sample_parameters()
            equilibrium_state = self.system.equilibrium(theta)
            x0 = self.sample_initial_condition(theta, scale=0.3)
            trajectory, success = self.simulate(theta, x0)
            
            if not success:
                failed_count+=1
                pbar.update(1)
                continue
 
            is_stable = self.check_stability(trajectory, theta)
            if is_stable:
                stable_count+=1
            else:
                unstable_count+=1

            dataset.append({ 'trajectory': trajectory, 'theta': theta, 'stable': is_stable, 'equilibrium': equilibrium_state, 'x0': x0.astype(np.float32), 'b1': float(theta[4]), 'b2': float(theta[5]) })
            pbar.update(1)
        
        pbar.close()
        
        print(f"Total trajectories: {len(dataset)}")
        print(f"Stable: {stable_count} ({stable_count/len(dataset):.1f})")
        print(f"Unstable: {unstable_count} ({unstable_count/len(dataset):.1f})")
        print(f"Failed: {failed_count}")
        
        output_path = Path(output_dir) / f"{output_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
    
        metadata = {
            'n_trajectories': len(dataset),
            'sim_time': self.sim_time,
            'dt': self.dt,
            'n_timesteps': self.n_timesteps,
            'stable_count': stable_count,
            'unstable_count': unstable_count,
            'param_ranges': { 'm1':[0.3, 0.7], 'm2':[0.3, 0.7], 'k1':[4.0, 8.0], 'k2':[4.0, 8.0], 'b1':[0.05, 0.5], 'b2':[0.05, 0.5]}
        }
        
        metadata_path = Path(output_dir) / f"{output_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return dataset
    
if __name__ == "__main__":
    gen = DataGenerator(sim_time=15.0, dt=0.1)
    dataset = gen.generate_dataset(n_trajectories=10000, output_dir='data',output_name='training_data_10k')
    
    sample = dataset[0]
    x_eq = gen.system.equilibrium(sample['theta'])
