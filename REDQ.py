import numpy as np
from networks import Actor, Critic, copy_weights
import torch
import torch.nn.functional as F
import os
    
class REDQ:
    """
    Implementation of Randomized Ensembled Double Q-learning (REDQ)
    """
    def __init__(self, state_dim, action_dim, max_action, 
                 gamma=0.99, tau=0.005, alpha=0.05, lr=1e-3, gradient_steps=1, 
                 batch_size=256,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        print(f'State Dimension: {state_dim}, Action Dimension: {action_dim}')
        
        self.gamma = gamma 
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.device = device
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # We instanciate two critic networks. We later use the minimum of the two Q-values
        # to try to reduce overestimation bias
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        
        # Create the target networks. They are not trained directly, 
        # but are used to compute the target Q-values. This provides a more stable training
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        
        # Initial hard update of the target networks (tau=1)
        copy_weights(self.critic_1_target, self.critic_1)
        copy_weights(self.critic_2_target, self.critic_2)
        
        # see that the target networks are the same as the critics
        assert all([torch.all(torch.eq(x, y)) for x, y in zip(self.critic_1_target.parameters(), self.critic_1.parameters())])
        assert all([torch.all(torch.eq(x, y)) for x, y in zip(self.critic_2_target.parameters(), self.critic_2.parameters())])
        
        
    
    def predict(self, state, deterministic=False):
        """
        Predict the action given the state
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        sampled_action, _, mean_action = self.actor.sample(state)
        
        action = mean_action if deterministic else sampled_action
        return action.detach().cpu().numpy()[0]
        
    
    def train(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return # If the replay buffer is not filled up, return
        
        for i in range(self.gradient_steps):
            # Randomly sample a batch of transitions from the replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
            
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.Tensor(done).to(self.device).unsqueeze(1)
            
            # Compute the target Q value 
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                
                q1_target = self.critic_1_target(next_state, next_action)
                q2_target = self.critic_2_target(next_state, next_action)
                
                y = reward + self.gamma * (1 - done) * (torch.min(q1_target, q2_target) - self.alpha * next_log_prob)
                
               
            # Update the Q functions (non-target)
            q1 = self.critic_1(state, action)
            q2 = self.critic_2(state, action)
            q1_loss = F.mse_loss(q1, y)
            q2_loss = F.mse_loss(q2, y)
            
            
            # Backpropagate the critics
            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            q1_loss.backward()
            q2_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()
            
            # Update the policy
            new_action, log_prob, _ = self.actor.sample(state)
            q1 = self.critic_1(state, new_action)
            q2 = self.critic_2(state, new_action)
            
            q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_prob - q).mean()
            
            # Backpropagate the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the target networks
            copy_weights(self.critic_1_target, self.critic_1, self.tau)
            copy_weights(self.critic_2_target, self.critic_2, self.tau)
            
    def save_checkpoint(self, filename):
        """
        Save the model
        """
        torch.save(self.actor.state_dict(), os.path.join(filename, "actor"))
        torch.save(self.critic_1.state_dict(), os.path.join(filename, "critic_1"))
        torch.save(self.critic_2.state_dict(), os.path.join(filename, "critic_2"))
        
        
    def load_checkpoint(self, filename):
        """
        Load the model
        """
        self.actor.load_state_dict(torch.load(os.path.join(filename, "actor")))
        self.critic_1.load_state_dict(torch.load(os.path.join(filename, "critic_1")))
        self.critic_2.load_state_dict(torch.load(os.path.join(filename, "critic_2")))
        
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
            
            
            
            
            
            
            
    
    