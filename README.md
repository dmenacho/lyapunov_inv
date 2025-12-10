# lyapunov_inv

## SETTINGS
The necessary libraries are listed in `requirements.txt`. Install CUDA depending on the NVIDIA processor available on your hardware. The respective data and model files are stored in a Google Drive folder; the link is provided in the paper document. The `src/` folder contains the paper, slides, and poster in PDF format.

## DATA GENERATION
Run `data_generation_lyap.py` to generate the simulated damped double-pendulum dynamical system.

## PRETRAINING
Run `surrogate.py` to obtain the weights for the DeepONet model trained with the generated data (`parametric_surrogate.pt`).  
Run `pinn_lyap.py` to obtain the weights for the Lyapunov-based PINN model trained with the stable data (`lyapunov_zubov.pt`).

## PARAMETER IDENTIFICATION
Run `inverse.py` to obtain the admissible parameters. The code produces two outputs:  
- `admissible_thetas_total.npy` — admissible parameters using the total loss  
- `admissible_thetas_data.npy` — admissible parameters without the Lyapunov loss

## VISUALIZATION
Run `visualization.ipynb` to visualize the frequency plots of the admissible parameters. This notebook loads the files used in the manuscript:  
- `admissible_thetas_data_<fig_pos>.npy`  
- `admissible_thetas_total_<fig_pos>.npy`

lyapunov_inv/
├── data_generation_lyap.py
├── inverse.py
├── pinn_lyap.py
├── surrogate.py
├── visualization.ipynb
├── requirements.txt
├── README.md
└── src/
    ├── DM Presentation.pdf
    ├── Final Draft_AI_for_Science_DMO.pdf
    └── Poster_AIforScience.pdf
