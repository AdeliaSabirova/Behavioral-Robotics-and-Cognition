import gym 


def environment_execution(env):
    env.reset()
    for _ in range(200):
        env.render()
        observation, rewars, done, info = env.step(env.action_space.sample())
    env.close()

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)



#Acrobot environment
print("Acrobot environment")
environment_execution(gym.make('Acrobot-v1'))

#CartPole environment
print("CartPole environment")
environment_execution(gym.make('CartPole-v1'))

#Pendulum environment
print("Pendulum environment")
environment_execution(gym.make('Pendulum-v0'))

#MountainCar environment
print("MountainCar environment")
environment_execution(gym.make('MountainCar-v0'))

#MountainCarContinuous environment
print("MountainCarContinuous environment")
environment_execution(gym.make('MountainCarContinuous-v0'))

