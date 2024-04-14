from matplotlib import pyplot as plt
import torch
import numpy as np
import random

def plot_stats(stats, output_dir):
    fig, axs = plt.subplots(3, 1, figsize=(5, 10))
    axs[0].plot(stats['score_history'])
    axs[0].set_title('Score history')
    axs[1].plot(stats['length_history'])
    axs[1].set_title('Length history')
    # Make a moving average of the last 100 episodes
    avg_score = [sum(stats['score_history'][i:i+100])/100 for i in range(len(stats['score_history'])-100)]
    axs[2].plot(avg_score)
    axs[2].set_title('Moving average of the score')
    plt.tight_layout()
    plt.savefig(output_dir)
    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    