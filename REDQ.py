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
                 gamma=0.99, tau=0.005, alpha=0.05, lr=1e-3,
                 batch_size=256, N=10, G=20, M=2,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print(f"Runing {self.__class__.__name__} with N={N}, G={G}, M={M}")
        
        self.gamma = gamma 
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        
        self.N = N # Number of critics
        self.G = G # Number of updates to the critics
        self.M = M # Number of indexes to sample from the critics
        
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # We instanciate the critics
        self.critics_list = []
        self.critic_optimizer_list = []
        self.critic_target_list = []
        for _ in range(self.N):
            critic = Critic(state_dim, action_dim).to(self.device)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
            target_critic = Critic(state_dim, action_dim).to(self.device)
            copy_weights(target_critic, critic)
            
            self.critics_list.append(critic)
            self.critic_optimizer_list.append(critic_optimizer)
            self.critic_target_list.append(target_critic)
            
            
        
        # see that the target networks are the same as the critics
        # do it for all the critics using for loop
        for i in range(1, self.N):
            assert all([torch.all(torch.eq(x, y)) for x, y in zip(self.critics_list[i].parameters(), self.critic_target_list[i].parameters())])
        
        
    
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
        
        for update in range(self.G):
            # Randomly sample a batch of transitions from the replay buffer
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
            
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.Tensor(done).to(self.device).unsqueeze(1)
            
            sample_indexes = np.random.choice(self.N, self.M, replace=False)
            # Compute the target Q value 
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                
                q_target_list = []
                for idx in sample_indexes:
                    q_target = self.critic_target_list[idx](next_state, next_action)
                    q_target_list.append(q_target)
                q_target = torch.cat(q_target_list, dim=1)
                min_q_target = torch.min(q_target, dim=1)[0].unsqueeze(1)
                                
                y = reward + self.gamma * (1 - done) * (min_q_target - self.alpha * next_log_prob)
                
                
            # Update the Q functions (non-target)
            q_a_list = []
            for i in range(self.N):
                q = self.critics_list[i](state, action)
                q_a_list.append(q)
                
            q_a_list = torch.cat(q_a_list, dim=1)
            y = y.repeat(1, self.N)
            
            q_loss = F.mse_loss(q_a_list, y) * self.N # Multiply by N to scale the loss
            
            for i in range(self.N):
                self.critic_optimizer_list[i].zero_grad()
            q_loss.backward()
            for i in range(self.N):
                self.critic_optimizer_list[i].step()    
                    
            # Update the target networks
            for i in range(self.N):
                copy_weights(self.critic_target_list[i], self.critics_list[i], self.tau)       
        
        
        # Update the policy
        new_action, log_prob, _ = self.actor.sample(state)
        q_ahat_list = []
        for i in range(self.N):
            self.critics_list[i].requires_grad_(False)
            q = self.critics_list[i](state, new_action)
            q_ahat_list.append(q)
        
        q_ahat_list = torch.cat(q_ahat_list, dim=1)
        mean_q_ahat = torch.mean(q_ahat_list, dim=1).unsqueeze(1)
        
        actor_loss = (self.alpha * log_prob - mean_q_ahat).mean()
        
        # Backpropagate the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for i in range(self.N):
            self.critics_list[i].requires_grad_(True)
             
        
            
    def save_checkpoint(self, filename):
        """
        Save the model
        """
        torch.save(self.actor.state_dict(), os.path.join(filename, "actor"))
        for i in range(self.N):
            torch.save(self.critics_list[i].state_dict(), os.path.join(filename, f"critic_{i}"))
                
        
    def load_checkpoint(self, filename):
        """
        Load the model
        """
        self.actor.load_state_dict(torch.load(os.path.join(filename, "actor")))
        for i in range(self.N):
            self.critics_list[i].load_state_dict(torch.load(os.path.join(filename, f"critic_{i}")))
        
        self.actor.eval()
        for i in range(self.N):
            self.critics_list[i].eval()

            
class SAC_REDQ(REDQ):
    def __init__(self, state_dim, action_dim, max_action, 
                 gamma=0.99, tau=0.005, alpha=0.05, lr=1e-3,
                 batch_size=256,
                 device='cpu'):
        N, G, M = 2, 1, 2
        super(SAC_REDQ, self).__init__(state_dim, action_dim, max_action, gamma, tau, alpha, lr, batch_size, N, G, M, device)

    