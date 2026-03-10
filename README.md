# lyapunov_inv

## SETTINGS
The necessary libraries are listed in `requirements.txt`. Install CUDA depending on the NVIDIA processor available on your hardware. The respective data and model files are stored in a Google Drive folder; the link is provided in the paper document. The `documentation/` folder contains the paper (version_1), slides, and poster in PDF format.

## FIRST VERSION
To evaluate the feasibility of the proposed research idea, I conducted experiments using a toy physical system consisting of a spring–damper model. The objective was to verify whether the inverse model could recover physically meaningful parameter values, such as a positive mass and a positive damping coefficient, which are required to guarantee system stability. The initial results were promising and encouraged further exploration of the approach, shifting the focus toward parameter identification in machine learning training dynamics.

Run `data_generation_lyap.py` to generate the simulated damped double-pendulum dynamical system.

Run `surrogate.py` to obtain the weights for the DeepONet model trained with the generated data (`parametric_surrogate.pt`).  
Run `pinn_lyap.py` to obtain the weights for the Lyapunov-based PINN model trained with the stable data (`lyapunov_zubov.pt`).

Run `inverse.py` to obtain the admissible parameters. The code produces two outputs:  
- `admissible_thetas_total.npy` — admissible parameters using the total loss  
- `admissible_thetas_data.npy` — admissible parameters without the Lyapunov loss

Run `visualization.ipynb` to visualize the frequency plots of the admissible parameters (Fig 4 and Fig 5). This notebook loads the files used in the manuscript:  
- `admissible_thetas_data_<fig_pos>.npy`  
- `admissible_thetas_total_<fig_pos>.npy`

Run `visualization_data.ipynb` to visualize some samples of the data (Fig 2)

Run `visualization_stability.ipynb` to visualize the stability of some samples (Fig 3)

## UPDATE VERSION

In this updated version, the goal is to identify parameter configurations that lead to stable neural network training. Initial experiments were performed using a simple neural network architecture (`inverse_nn_weight.py`, `pinn_lyap_nn.py`) as well as a ResNet-based model (`inverse_resnet_weight.py`, `pinn_lyap_resnet.py`) for the CIFAR10 classification task. The results and admissible parameter regions can be visualized using the notebook `test_admissible.ipynb`

The final version integrates the entire pipeline into a single file, `full_pipeline.py`, which uses libraries created in the `src/` folder. The file `learners.py` implements the Lyapunov-based training with NMIST or CIFAR10 datasets, `inverse_modeling.py` describes the weight initialization identification process, and `dynamic_models.py` contains the reduced deep learning architectures used in the experiments (ResNet, AlexNet, VGG, EfficientNet, ConvNeXt, and UNet)

```bash
full_pipeline.py full --num_train_samples 1000 --num_trajectories 500 --traj_steps 50 --gd_lr 0.01 --constraint_threshold 0.1 --dataset MNIST  --test_comparative_epochs 50 --test_comparative_lr 0.1 --target_model_name TinyResNet --set_normalization False
```

## DOCUMENT ORGANIZATION

lyapunov_inv/
    
    ├── InvLyapunov_FirstVersion/
    
        ├── data_generation_lyap.py
        ├── inverse.py
        ├── pinn_lyap.py
        ├── surrogate.py
        └── visualization.ipynb

    ├── Docummentation/
        ├── DM Presentation.pdf
        ├── Final Draft_AI_for_Science_DMO.pdf
        └── Poster_AIforScience.pdf

    ├── src/
        ├── dynamics_models.py
        ├── inverse_modeling.py
        ├── learners.py
        ├── lyap_models.py
        └── vis_utils.pdf
        
    ├── full_pipeline.py
    ├── test_admissible.ipynb
    ├── requirements.txt
    └── README.md
