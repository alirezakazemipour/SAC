import gym
from agent import SAC

ENV_NAME = "Pendulum-v0"
test_env = gym.make(ENV_NAME)

n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bounds = [test_env.action_space.high[0], test_env.action_space.low[0]]

MAX_EPISODES = 100
memory_size = 100
batch_size = 256

if __name__ == "__main__":
    print(f"Number of states:{n_states}\n"
          f"Number of actions:{n_actions}\n"
          f"Action boundaries:{action_bounds}")

    env = gym.make(ENV_NAME)
    agent = SAC(n_states=n_states,
                n_actions=n_actions,
                memory_size=memory_size,
                batch_size=batch_size)

    for episode in range(MAX_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, reward, done, action, next_state)
            loss = agent.train()
            if done:
                break
            state = next_state



