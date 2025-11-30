import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from ActorCritic import ActorCritic

#paramaters

num_episodes = 1000
gamma = 0.9
actor_lr = 1e-3
critic_lr = 1e-2
n_hiddens = 16
env_name = "CartPole-v1"
return_list =[]
env = gym.make(env_name)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = ActorCritic(n_states, n_hiddens, n_actions, actor_lr, critic_lr, gamma)
for i in range(num_episodes):
    state = env.reset()[0]
    done = False
    episode_return = 0

    transition_dict={
        'states':[],
        'actions':[],
        'next_states':[],
        'rewards':[],
        'dones':[]
    }
    while not done:
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)

        state = next_state
        episode_return += reward

    return_list.append(episode_return)
    agent.update(transition_dict)
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
plt.plot(return_list)
plt.show()