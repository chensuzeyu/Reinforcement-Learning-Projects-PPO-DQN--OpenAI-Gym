import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym.wrappers import RecordVideo
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')  # 使用TkAgg后端确保跨平台兼容性

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 神经网络模块
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        
    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)
    
    def get_action(self, state):
        with torch.no_grad():
            logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# 经验回放缓冲区
class ExperienceBuffer:
    def __init__(self, capacity, state_dim):
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.terminateds = np.zeros(capacity, dtype=bool)
        self.truncateds = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.capacity = capacity
        
    def store(self, state, action, reward, value, log_prob, terminated, truncated):
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.terminateds[idx] = terminated
        self.truncateds[idx] = truncated
        self.ptr += 1
        
    def get(self):
        return (
            torch.tensor(self.states[:self.ptr], device=device),
            torch.tensor(self.actions[:self.ptr], device=device),
            torch.tensor(self.rewards[:self.ptr], device=device),
            torch.tensor(self.values[:self.ptr], device=device),
            torch.tensor(self.log_probs[:self.ptr], device=device),
            self.terminateds[:self.ptr],
            self.truncateds[:self.ptr]
        )
    
    def clear(self):
        self.ptr = 0

# PPO训练器
class PPOTrainer:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.model = ActorCritic(env.observation_space.shape[0], 
                                env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.buffer = ExperienceBuffer(args.batch_size, env.observation_space.shape[0])
        
    def compute_returns_advantages(self, last_value=0):
        states, actions, rewards, values, log_probs, terminateds, truncateds = self.buffer.get()
        with torch.no_grad():
            _, next_value = self.model(states[-1].unsqueeze(0))
        
        returns = np.zeros(self.buffer.ptr, dtype=np.float32)
        advantages = np.zeros(self.buffer.ptr, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(self.buffer.ptr)):
            if t == self.buffer.ptr - 1:
                next_non_terminal = 1.0 - (terminateds[t] | truncateds[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - (terminateds[t+1] | truncateds[t+1])
                next_value = values[t+1]
            
            delta = rewards[t] + self.args.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + values[t]
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        states, actions, _, _, old_log_probs, _, _ = self.buffer.get()
        returns, advantages = self.compute_returns_advantages()
        returns = torch.tensor(returns, device=device)
        advantages = torch.tensor(advantages, device=device)
        
        # 训练多个epoch
        for _ in range(self.args.epochs):
            indices = torch.randperm(self.buffer.ptr, device=device)
            for start in range(0, self.buffer.ptr, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                idx = indices[start:end]
                
                # 计算新策略
                logits, values = self.model(states[idx])
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions[idx])
                
                # 重要性采样比率
                ratio = (new_log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * (values - returns[idx]).pow(2).mean()
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # 梯度裁剪
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
        self.buffer.clear()
        return loss.item(), entropy.item()

# 实时可视化模块
class TrainingVisualizer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 9))
        self.rewards = []
        self.losses = []
        self.entropies = []
        
    def update(self, reward, loss, entropy):
        self.rewards.append(reward)
        self.losses.append(loss)
        self.entropies.append(entropy)
        
        self.ax1.clear()
        self.ax1.plot(self.rewards, color='tab:blue')
        self.ax1.set_title('Episode Reward')
        
        self.ax2.clear()
        self.ax2.plot(self.losses, color='tab:orange')
        self.ax2.set_title('Training Loss')
        
        self.ax3.clear()
        self.ax3.plot(self.entropies, color='tab:green')
        self.ax3.set_title('Policy Entropy')
        
        plt.tight_layout()
        plt.pause(0.05)

# 训练流程
def train(args):
    env = gym.make('CartPole-v1')
    if args.record:
        env = RecordVideo(env, 'video', episode_trigger=lambda x: x % 50 == 0)
    
    trainer = PPOTrainer(env, args)
    visualizer = TrainingVisualizer()
    
    episode = 0
    total_reward = 0
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)
    
    while episode < args.max_episodes:
        # 收集经验
        for _ in range(args.batch_size):
            action, log_prob = trainer.model.get_action(state)
            value = trainer.model(state)[1].item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trainer.buffer.store(state.cpu().numpy(), action, reward, 
                               value, log_prob.item(), terminated, truncated)
            
            state = torch.FloatTensor(next_state).to(device)
            total_reward += reward
            
            if done:
                episode += 1
                print(f"Episode {episode}: Reward {total_reward}")
                
                # 更新可视化
                loss, entropy = trainer.update()
                visualizer.update(total_reward, loss, entropy)
                
                total_reward = 0
                state, _ = env.reset()
                state = torch.FloatTensor(state).to(device)
                
    env.close()
    torch.save(trainer.model.state_dict(), 'ppo_cartpole.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--max_episodes', type=int, default=300)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    
    train(args)