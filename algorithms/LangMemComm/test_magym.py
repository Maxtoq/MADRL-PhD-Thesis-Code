import gym

env = gym.make('ma_gym:Switch2-v0')
print(env.action_space)
print(env.observation_space)
exit()
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    ep_reward += sum(reward_n)
    print(obs_n)
    # time.sleep(0.01)
env.close()