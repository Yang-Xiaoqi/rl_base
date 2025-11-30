import gym
import numpy as np

env = gym.make('CartPole-v1')
def discretize_state(observation):
    # 定义每个观察维度的区间数量
    cart_position_bins = np.linspace(-2.4, 2.4, 10)
    cart_velocity_bins = np.linspace(-4, 4, 10)
    pole_angle_bins = np.linspace(-0.2, 0.2, 10)
    pole_velocity_bins = np.linspace(-4, 4, 10)

    # 获取离散化索引
    cart_position, cart_velocity, pole_angle, pole_velocity = observation

    # 使用digitize函数将连续值映射到离散区间
    discretized = [
        np.digitize(cart_position, cart_position_bins),
        np.digitize(cart_velocity, cart_velocity_bins),
        np.digitize(pole_angle, pole_angle_bins),
        np.digitize(pole_velocity, pole_velocity_bins)
    ]

    # 转换为元组，作为Q表的键
    return tuple(discretized)

q_table = {}

def get_q_value(state, action):
    if state not in q_table:
        q_table[state] = np.zeros(env.action_space.n)
    return q_table[state][action]
def update_q_table(state, action, reward, next_state,done):
    learning_rate = 0.1
    discount_factor = 0.99

    if done:
        q_table[state][action] = (q_table[state][action] + learning_rate * reward -learning_rate * q_table[state][action])
    else:
        next_max_q = np.max([get_q_value(next_state, a) for a in range(env.action_space.n)])
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + \
                                  learning_rate * (reward + discount_factor * next_max_q)

num_episodes = 1000
for episode in range(num_episodes):
    observation, _ = env.reset()
    state = discretize_state(observation)
    total_reward = 0
    done = False

    exploration_rate = max(0.01,0.1-0.01*(episode/200))

    while not done:
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax([get_q_value(state, action) for action in range(env.action_space.n)])

        observation,reward,done,_,_ = env.step(action)
        next_state = discretize_state(observation)
        update_q_table(state, action, reward, next_state,done)
        state = next_state
        total_reward += reward
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")



# 测试
env = gym.make('CartPole-v1')
observation, _ = env.reset()
done = False
total_reward = 0

while not done:
    state = discretize_state(observation)
    # 选择Q值最大的动作
    action = np.argmax([get_q_value(state, a) for a in range(env.action_space.n)])

    observation, reward, done, _, info = env.step(action)
    total_reward += reward

print(f"测试结束，总奖励: {total_reward}")
env.close()