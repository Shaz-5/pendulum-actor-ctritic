import os
from collections import deque
from itertools import count
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from IPython import display

from networks.actor_critic import Actor, Critic
from utils.replay import ReplayBuffer
from utils.ou_noise import OUNoise

class DDPGTrainer:
    
    def __init__(self, env, agent_name, buffer_size=int(1e5), batch_size=128, gamma=0.99, tau=1e-3,
                 alpha=1e-4, beta=1e-3, weight_decay=0, seed=42, model_load_path=None, model_save_path=None):
        
        self.env = env
        self.env.reset()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.seed = random.seed(seed)
        
        # hyperparameters
        self.buffer_size = buffer_size     # replay buffer size
        self.batch_size = batch_size       # minibatch size
        self.gamma = gamma                 # discount factor
        self.tau = tau                     # for soft update of target parameters
        self.alpha = alpha                 # learning rate of the actor
        self.beta = beta                   # learning rate of the critic
        self.weight_decay = weight_decay   # L2 weight decay
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent_name = agent_name
        
        if model_save_path is None:
            default_model_save_path = f'./models/{agent_name}'
            self.model_save_path = default_model_save_path
            if not os.path.exists(default_model_save_path):
                os.makedirs(default_model_save_path)
        else:
            self.model_save_path = model_save_path
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
        
        # Actor
        self.actor = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)

        # Critic
        self.critic = Critic(self.state_size, self.action_size, seed).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.beta, weight_decay=self.weight_decay)
        
        # Noise
        self.noise = OUNoise(self.action_size, seed=seed)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.action_size, self.batch_size, seed)
        
        # Load pre-trained models
        self.is_trained = False
        if model_load_path:
            try:
                self.is_trained = True
                self.actor.load_state_dict(torch.load(model_load_path+'/actor.pth'))
                self.critic.load_state_dict(torch.load(model_load_path+'/critic.pth'))
            except (FileNotFoundError, torch.cuda.CudaError, RuntimeError, KeyError) as e:
                print(f"Error loading pre-trained model: {e}")
    
    
    def select_action(self, state, add_noise=True):
        """
        Select actions for the given state based on the current policy.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.actor.eval()  # Set the actor network to evaluation mode
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()
        self.actor.train()  # Set the actor network back to training mode
        if add_noise:
            action += self.noise.sample()  # Add noise to the action if required (for exploration)
        return np.clip(action, -1, 1)  # Clip the action to be within the valid range [-1, 1]
    
    
    def soft_update(self, source_model, target_model, tau=None):
        """
        Update the weights of target netowrks using soft update rule.
        
        θ_target = τ * θ_source + (1 - τ) * θ_target
        """
        if tau is None:
            tau = self.tau
            
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
            
            
    def optimize_model(self):
        """
        Update policy and value parameters using a batch of experiences.
        """
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Update Critic model
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        
        Q_targets = rewards + (self.gamma * next_Q_targets * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # Update Actor model
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target network models
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)
        
    
    def ddpg_train(self, n_episodes=1000, max_steps=300, print_every=100, plot_save_path=None):
        """
        Train the agent using DDPG.
        """
        scores_window = deque(maxlen=print_every)
        all_scores = []
        
        for episode in tqdm(range(1, n_episodes + 1), desc="Training agent using DDPG.."):
            state, _ = self.env.reset()
            # print(state)
            self.noise.reset()
            score = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                self.memory.push(state, action, reward, next_state, done)
                self.optimize_model()
                
                state = next_state
                score += reward
                
                if done:
                    break
                
            all_scores.append(score)
            scores_window.append(score)
            
            if episode % print_every == 0:
                avg_score = np.mean(scores_window)
                print(f'\rEpisode {episode}\tAverage Score: {avg_score}')

                # Save model checkpoints
                torch.save(self.actor.state_dict(), self.model_save_path+'/actor.pth')
                torch.save(self.critic.state_dict(), self.model_save_path+'/critic.pth')
                
        # Plot training performance
        self.plot_scores(scores=all_scores, plot_save_path=plot_save_path)
        
        return all_scores, scores_window
    
    
    def test_model(self, max_steps=300, render_save_path=None):
        """
        Run the environment using trained model.
        """
        episode_reward = 0

        state, _ = self.env.reset()

        images = []

        with torch.inference_mode():
            for t in range(max_steps):
                if render_save_path:
                    img = self.env.render()
                    images.append(img)

                action = self.select_action(state, add_noise=False)
                state, reward, done, _, _ = self.env.step(action)

                episode_reward += reward

                if done:
                    break

        if render_save_path:
            imageio.mimsave(f'{render_save_path}.gif', images, fps=30, loop=0)
            self.env.close()
            with open(f'{render_save_path}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))

        return episode_reward
    
    
    def plot_scores(self, scores, plot_save_path):
        """
        Plot performance of agent.
        """
        plt.figure(figsize=(10,8))
        plt.plot(scores)
        plt.title(f'Performance of {self.agent_name}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()