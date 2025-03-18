# pip install swig
# pip install box2d box2d-kengz --user
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.fc(state)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.buffer = ReplayBuffer(10000)
        
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1])).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(batch[2]))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(np.array(batch[4]))
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 训练函数
def train(env, agent, episodes=500):
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            # 显示游戏窗口
            env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.buffer.push((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            
            agent.update_model()
            
            if done or truncated:
                break
        
        # 定期更新目标网络
        if episode % 10 == 0:
            agent.update_target()
        
        print(f"Episode: {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

# 测试函数（显示完整游戏过程）
def test(env, agent):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        env.render()  # 显示游戏窗口
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"Test Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # 创建环境（设置render_mode为"human"以显示窗口）
    env = gym.make('LunarLander-v2', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # 训练智能体
    train(env, agent, episodes=200)
    
    # 测试训练好的智能体
    test(env, agent)
    
    env.close()