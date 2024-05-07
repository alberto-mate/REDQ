import gymnasium as gym
import numpy as np
import torch
import os
import json
from gymnasium.wrappers import RecordVideo
import time

from replay_memory import ReplayMemory
from utils import plot_stats, set_seed
from REDQ import REDQ

from args import OPT as opt
### Configuration ###
render = 'rgb_array' # 'human', 'rgb_array' or 'none'
env_name = opt.env
load_checkpoint = None # f'outputs/{env_name}/experiment_1'
total_timesteps = opt.total_timesteps
learning_starts = 5_000
seed = opt.seed
exp_name = opt.exp_name
kwargs = opt.kwargs if opt.kwargs is not None else {}
#####################



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
evaluate = load_checkpoint is not None
set_seed(seed)
print(f"Rendering with {render} mode")
env_kwargs = dict(continuous=True) if 'LunarLander' in env_name else {}
env = gym.make(env_name, render_mode=render, **env_kwargs)
env.action_space.seed(seed)
memory = ReplayMemory(capacity=100_000, seed=seed)
agent = REDQ(state_dim=env.observation_space.shape[0], 
            action_dim=env.action_space.shape[0],
            max_action=env.action_space.high[0],
            device=device, **kwargs)



if not evaluate:
    output_dir = f'outputs/{env_name}/{exp_name}'
    id_experiment = len(os.listdir(output_dir)) + 1 if os.path.exists(output_dir) else 1
    output_dir = os.path.join(output_dir, f'experiment_{id_experiment}')
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = load_checkpoint
    agent.load_checkpoint(output_dir)
    print(f'Loaded checkpoint from {output_dir}, evaluating...')
    
if render == 'rgb_array':
    video_name = 'video_eval' if evaluate else 'video_train'
    record_freq = 1 if evaluate else 40
    # Record video every 40 episodes during training 
    env = RecordVideo(env, os.path.join(output_dir, video_name), disable_logger=True, episode_trigger=lambda x: x % record_freq == 0)
    print(f"Recording video in {output_dir}")
    
    
current_timestep = 0
n_episodes = 0
score_history = [] # Score history
length_history = [] # Length history 
timesteps_history = [] # Timesteps history
best_score = -np.inf
start_time = time.time()
while current_timestep < total_timesteps:
    n_episodes += 1
    obs, _ = env.reset(seed=seed+n_episodes) # Reset the environment with the seed
    done = False
    score = 0
    length_episode = 0
    while not done:
        if current_timestep <= learning_starts:
            action = env.action_space.sample() # Sample random action during the initial learning phase
        else:
            with torch.no_grad():
                action = agent.predict(obs, deterministic=evaluate) # Predict the action using the actor network
                
        # Perform the action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store the transition in the replay memory
        memory.push(obs, action, reward, next_obs, done)
        if not evaluate and current_timestep > learning_starts:
            agent.train(memory)
        
        score += reward
        length_episode += 1
        obs = next_obs
        
        current_timestep += 1
    
    env.close()  
    
    score_history.append(score)  
    length_history.append(length_episode)  
    timesteps_history.append(current_timestep)
    avg_score = np.mean(score_history[-100:])   
    if avg_score > best_score:
        best_score = avg_score
        if not evaluate:
            agent.save_checkpoint(output_dir)
    
    # Log the stats
    if n_episodes % 1 == 0:
        print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}]", end=' ')
        print(f'Ep: {n_episodes} | t: {current_timestep}/{total_timesteps} | Score: {score:.2f} | Avg Score: {avg_score:.2f}')
        
        stats = dict(best_score=best_score, score_history=score_history, length_history=length_history, timesteps_history=timesteps_history)
        json.dump(stats, open(os.path.join(output_dir, 'stats.json'), 'w'))
        
    
# Save the stats as json
stats = dict(best_score=best_score, score_history=score_history, length_history=length_history, timesteps_history=timesteps_history)
json.dump(stats, open(os.path.join(output_dir, 'stats.json'), 'w'))

plot_stats(stats, os.path.join(output_dir, 'stats.png'))

# Rename output directory to include the best score
new_output_dir = f'{output_dir}_{int(best_score)}'
os.rename(output_dir, new_output_dir)
