#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=run
#SBATCH --output=./job-outs/WZVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.0_sigma_Vtilde_0.1/chiUnderline_0.2_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_1.5_psi_h_1.5/trial_4_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_5_iter_num_15/run.out
#SBATCH --error=./job-outs/WZVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.0_sigma_Vtilde_0.1/chiUnderline_0.2_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_1.5_psi_h_1.5/trial_4_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_5_iter_num_15/run.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

module load python/booth/3.8/3.8.5 

srun python3 NN_structure.py    --XiE_layers 3 --XiH_layers 3 --kappa_layers 3
srun python3 standard_BFGS.py   --chiUnderline 0.2 --a_e 0.15 --a_h 0.05 --gamma_e 1.0 --gamma_h 3.0 --psi_e 1.5 --psi_h 1.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.1 --XiE_layers 3 --XiH_layers 3 --kappa_layers 3 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 5 --iter_num 15 --trial 4 --chi_position_tolerance 0.0 --chi_value_tolerance 0.0 --chi_max_iterations 500 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14

