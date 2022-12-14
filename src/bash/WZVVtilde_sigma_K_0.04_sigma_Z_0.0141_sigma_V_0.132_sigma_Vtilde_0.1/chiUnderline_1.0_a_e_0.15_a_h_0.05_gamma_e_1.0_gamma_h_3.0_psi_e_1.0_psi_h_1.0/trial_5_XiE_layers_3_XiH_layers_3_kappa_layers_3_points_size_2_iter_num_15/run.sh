#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=run
#SBATCH --output=./job-outs/WZVVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.132_sigma_Vtilde_0.1/chiUnderline_1.0_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_1.0_psi_h_1.0/trial_5_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_2_iter_num_15/run.out
#SBATCH --error=./job-outs/WZVVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.132_sigma_Vtilde_0.1/chiUnderline_1.0_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_1.0_psi_h_1.0/trial_5_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_2_iter_num_15/run.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=56G

module load python/anaconda-2021.05

python3 NN_structure.py    --XiE_layers 3 --XiH_layers 3 --kappa_layers 3
python3 standard_BFGS.py   --chiUnderline 1.0 --a_e 0.15 --a_h 0.05 --gamma_e 1.0 --gamma_h 3.0 --psi_e 1.0 --psi_h 1.0 --nWealth 20 --nZ 10 --nV 10 --nVtilde 10 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.132 --sigma_Vtilde_norm 0.1 --XiE_layers 3 --XiH_layers 3 --kappa_layers 3 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 15 --trial 5 --chi_position_tolerance 0.0 --chi_value_tolerance 0.0 --chi_max_iterations 500 --W_fix 14 --Z_fix 4 --V_fix 4 --Vtilde_fix 4

