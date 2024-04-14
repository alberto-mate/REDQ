# Description: Run the experiments for the paper.

## LunarLander-v2
# echo "Running experiments for LunarLander-v2"
# SAC with alpha = 0.05
# echo "Running SAC with alpha = 0.05"
# python main.py --env LunarLander-v2 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 1
# python main.py --env LunarLander-v2 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 2
# python main.py --env LunarLander-v2 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 3
# python main.py --env LunarLander-v2 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 4

# REDQ with alpha = 0.05 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.05 and N=5, G=5, M=2"
python main.py --env LunarLander-v2 --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 1
python main.py --env LunarLander-v2 --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 2

# REDQ with alpha = 0.2 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.2 and N=5, G=5, M=2"
python main.py --env LunarLander-v2 --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 1
python main.py --env LunarLander-v2 --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 2

## BipedalWalker-v3
echo "Running experiments for BipedalWalker-v3"
# SAC with alpha = 0.05
echo "Running SAC with alpha = 0.05"
python main.py --env BipedalWalker-v3 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 1
python main.py --env BipedalWalker-v3 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 2
python main.py --env BipedalWalker-v3 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 3
python main.py --env BipedalWalker-v3 --kwargs N=2 G=1 M=2 alpha=0.05 --exp_name SAC_alpha0.05 --total_timesteps 200000 --seed 4

# REDQ with alpha = 0.05 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.05 and N=5, G=5, M=2"
python main.py --env BipedalWalker-v3 --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 1
python main.py --env BipedalWalker-v3 --kwargs N=5 G=5 M=2 alpha=0.05 --exp_name REDQ_alpha0.05_N5_G5_M2 --total_timesteps 200000 --seed 2

# REDQ with alpha = 0.2 and N=5, G=5, M=2
echo "Running REDQ with alpha = 0.2 and N=5, G=5, M=2"
python main.py --env BipedalWalker-v3 --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 1
python main.py --env BipedalWalker-v3 --kwargs N=5 G=5 M=2 alpha=0.2 --exp_name REDQ_alpha0.2_N5_G5_M2 --total_timesteps 200000 --seed 2