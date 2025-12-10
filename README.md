# lyapunov_inv

# lyapunov_inv

## SETTINGS
The necessary libraries are listed in `requirements.txt`.  
Install CUDA depending on the NVIDIA processor available on your hardware.

The respective data and model files are stored in a Google Drive folder;  
the link is provided in the paper document.

The `src/` folder contains the paper, slides, and poster in PDF format.

---

## DATA GENERATION
Run the following script to generate the simulated damped double-pendulum dynamical system:

```bash
python data_generation_lyap.py
PRETRAINING
1. Surrogate Model (DeepONet)
Run the following script to obtain the weights for the DeepONet model trained with the generated data:

bash
Copiar código
python surrogate.py
This will generate:

Copiar código
parametric_surrogate.pt
2. Lyapunov-based PINN
Run the following script to obtain the weights for the Lyapunov-based PINN model trained with the stable trajectories:

bash
Copiar código
python pinn_lyap.py
This will generate:

Copiar código
lyapunov_zubov.pt
PARAMETER IDENTIFICATION
Run the inverse model to compute the admissible parameters:

bash
Copiar código
python inverse.py
The script generates two output files:

admissible_thetas_total.npy — admissible parameters using the total loss

admissible_thetas_data.npy — admissible parameters without the Lyapunov loss

VISUALIZATION
To visualize the frequency plots of the admissible parameters, run the notebook:

Copiar código
visualization.ipynb
This notebook loads the following files used in the manuscript:

admissible_thetas_data_<fig_pos>.npy

admissible_thetas_total_<fig_pos>.npy

yaml
Copiar código
