import gym

env = gym.make('CartPole-v0')

# for 1000 episodes
for episode in range(1000):
    # reset environment and total reward
    observation = env.reset()
    total_reward = 0


    for t in range(300):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print "Reward : " + str(total_reward)
            total_reward = 0
            break