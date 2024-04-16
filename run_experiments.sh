#!/bin/bash

# Description: Run the experiments for the paper.
## LunarLander-v2
echo "Running experiments for LunarLander-v2"
# SAC with alpha = 0.05
echo "Running SAC with alpha = 0.05"
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 1 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 2 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 3 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 4 --env LunarLander-v2

# # REDQ with alpha = 0.05 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.05 and N=5, G=5, M=2"
python main.py --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 1 --env LunarLander-v2
python main.py --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 2 --env LunarLander-v2

# REDQ with alpha = 0.2 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.2 and N=5, G=5, M=2"
python main.py --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 1 --env LunarLander-v2
python main.py --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 2 --env LunarLander-v2

# SAC with alpha = 0.2
echo "Running SAC with alpha = 0.2"
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 200000 --seed 42 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 200000 --seed 43 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 200000 --seed 44 --env LunarLander-v2
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 200000 --seed 45 --env LunarLander-v2

# REDQ with alpha = 0.05 and N=3, G=10, M=2
echo "Running REDQ with alpha = 0.05 and N=3, G=10, M=2"
python main.py --kwargs N=3 G=10 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N3_G10_M2 --total_timesteps 200000 --seed 321 --env LunarLander-v2
python main.py --kwargs N=3 G=10 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N3_G10_M2 --total_timesteps 200000 --seed 32 --env LunarLander-v2

## BipedalWalker-v3
echo "Running experiments for BipedalWalker-v3"
# SAC with alpha = 0.05
echo "Running SAC with alpha = 0.05"
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 500000 --seed 1 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 500000 --seed 2 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 500000 --seed 3 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 500000 --seed 4 --env BipedalWalker-v3

# REDQ with alpha = 0.05 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.05 and N=5, G=5, M=2"
python main.py --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 500000 --seed 1 --env BipedalWalker-v3
python main.py --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 500000 --seed 2 --env BipedalWalker-v3

# REDQ with alpha = 0.2 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.2 and N=5, G=5, M=2"
python main.py --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 500000 --seed 1 --env BipedalWalker-v3
python main.py --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 500000 --seed 2 --env BipedalWalker-v3

# SAC with alpha = 0.2
echo "Running SAC with alpha = 0.2"
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 500000 --seed 42 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 500000 --seed 43 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 500000 --seed 44 --env BipedalWalker-v3
python main.py --kwargs N=2 G=1 M=2 alpha=0.2 --exp_name SAC_alpha0.2 --total_timesteps 500000 --seed 45 --env BipedalWalker-v3

# REDQ with alpha = 0.05 and N=3, G=10, M=2
echo "Running REDQ with alpha = 0.05 and N=3, G=10, M=2"
python main.py --kwargs N=3 G=10 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N3_G10_M2 --total_timesteps 500000 --seed 321 --env BipedalWalker-v3
python main.py --kwargs N=3 G=10 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N3_G10_M2 --total_timesteps 500000 --seed 32 --env BipedalWalker-v3