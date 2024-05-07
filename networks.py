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
    """
    Critic network to evaluate the Q-value of the state-action pair
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # We instanciate the critic network with 2 hidden layers with 256 units each
        # The input size is the sum of the state and action dimensions and the expected output is 1
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        weights_init_(self) # Weights initialization with Xavier uniform
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Actor(nn.Module):
    """
    Actor network to predict the mean and log_std of the action distribution
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # The actor network has 1 hidden layer with 256 units
        # The input size is the state dimension and the output size is the action dimension where
        # the mean and log_std of the action distribution are predicted
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
        mu, log_sigma = self.forward(state) # Get mean and log_std from the network
        sigma = log_sigma.exp()
        dist = torch.distributions.Normal(mu, sigma) # Create a normal distribution with the mean and std
        action = dist.rsample() # Sample an action from the distribution with reparametrization trick
        log_prob = dist.log_prob(action) # Compute the log probability of the action

        # Enforcing action bounds. For the tanh activation function, the action bounds are -1 and 1
        # so we adjust the action and log_prob accordingly. For a clear explanation, see the appendix of the paper
        adjusted_action = torch.tanh(action) * self.max_action
        adjusted_log_prob = log_prob - torch.log(self.max_action * (1-torch.tanh(action).pow(2)) + epsilon)
        adjusted_log_prob = adjusted_log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mu) * self.max_action # Compute the mean action from the tanh activation function for deterministic policy
        
        return adjusted_action, adjusted_log_prob, mean
        
        

def copy_weights(target, source, tau=1):
    """
    Copy the weights from the source network to the target network using the formula:
        target_weights = tau * source_weights + (1 - tau) * target_weights
    
    If tau=1, the weights are copied exactly. If tau=0, the weights are not copied.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)