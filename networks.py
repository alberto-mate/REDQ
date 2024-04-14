import torch
from torch import nn
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        weights_init_(self)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_sigma = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
        weights_init_(self)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX) # Clamping the sigma value to avoid numerical instability
        
        return mu, log_sigma
    
    def sample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        # Enforcing action bounds
        adjusted_action = torch.tanh(action) * self.max_action
        adjusted_log_prob = log_prob - torch.log(self.max_action * (1-torch.tanh(action).pow(2)) + epsilon)
        adjusted_log_prob = adjusted_log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mu) * self.max_action
        
        return adjusted_action, adjusted_log_prob, mean
        
        

def copy_weights(target, source, tau=1):
    """
    Copy the weights from the source network to the target network using the formula:
        target_weights = tau * source_weights + (1 - tau) * target_weights
    
    If tau=1, the weights are copied exactly. If tau=0, the weights are not copied.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)