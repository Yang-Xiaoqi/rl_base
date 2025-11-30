import gym
env = gym.make('CartPole-v0')
state = env.reset()
print(state)
print(env.action_space)
print(env.observation_space)
for t in range(100):
    print(state)
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()