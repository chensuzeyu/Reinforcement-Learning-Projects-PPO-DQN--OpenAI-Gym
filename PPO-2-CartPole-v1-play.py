import gym
import torch
import time
from torch.distributions import Categorical
import argparse

# 加载训练好的模型
def load_model(model_path, state_dim=4, action_dim=2):
    model = ActorCritic(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model

# 可视化游戏过程
def play(env, model, num_episodes=3, render_delay=0.02):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            with torch.no_grad():  # 禁用梯度计算
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, _ = model(state_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 渲染环境并添加延迟
            env.render()
            time.sleep(render_delay)  # 控制渲染速度
            
            total_reward += reward
            state = next_state
        
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
        time.sleep(1)  # 每局结束暂停1秒
    
    env.close()

# 网络结构定义（与训练代码保持一致）
class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU()
        )
        self.actor = torch.nn.Linear(256, action_dim)
        self.critic = torch.nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ppo_cartpole.pth",
                      help="训练好的模型路径")
    parser.add_argument("--episodes", type=int, default=3,
                      help="要演示的回合数")
    parser.add_argument("--speed", type=float, default=1.0,
                      help="游戏速度倍数（>1加速，<1减速）")
    args = parser.parse_args()

    # 创建环境（必须使用与训练相同的环境版本）
    env = gym.make('CartPole-v1', render_mode="human")
    
    try:
        # 加载模型
        model = load_model(args.model)
        print("模型加载成功！开始演示...")
        
        # 计算渲染延迟（默认0.02对应正常速度）
        play(env, model, 
            num_episodes=args.episodes,
            render_delay=0.02/args.speed)
            
    except KeyboardInterrupt:
        print("用户中断演示")
    finally:
        env.close()