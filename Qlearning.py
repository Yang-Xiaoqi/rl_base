import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):
        # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        # Q网络
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):
        # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Q值
        q_values = self.q_net(states).gather(1, actions)
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播更新参数
        dqn_loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    def save(self, path):
        """保存模型参数"""
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """加载模型参数"""
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()


    # 超参数设置
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 256

# 设备选择
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 环境初始化
env_name = 'CartPole-v1'
env = gym.make(env_name)

# 随机种子设置
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)

# 实例化经验回放池
replay_buffer = ReplayBuffer(buffer_size)

# 获取状态和动作维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 实例化DQN智能体
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

return_list = []

# 训练循环
# 外层循环用于分块显示进度条 (共10个大块)
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            # 重置环境
            reset_data = env.reset(seed=0)
            # gymnasium 返回 (observation, info)，需解包
            state = reset_data[0]
            done = False

            while not done:
                # 选择动作
                action = agent.take_action(state)
                # 执行动作
                step_result = env.step(action)
                next_state, reward, done, truncated, _ = step_result
                # 处理终止条件 (done 或 truncated 都视为结束)
                done = done or truncated

                # 存入经验回放池
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

            return_list.append(episode_return)

            # 每10个episode打印一次平均回报
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# 保存模型参数
model_path = 'dqn_cartpole.pth'
agent.save(model_path)
print(f"模型已保存到 {model_path}")

# 绘制原始回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# 测试模型
agent.load(model_path)

test_episodes = 10
test_returns = []

for i in range(test_episodes):
    state, _ = env.reset(seed=i)  # 不同种子方便观察
    done = False
    episode_return = 0

    while not done:
        action = agent.take_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        state = next_state
        episode_return += reward

    test_returns.append(episode_return)
    print(f"Test Episode {i+1}: return = {episode_return}")

print(f"平均测试回报: {np.mean(test_returns)}")

