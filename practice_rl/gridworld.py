import gymnasium as gym

# Create the Mountain Car environment (discrete version)
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# Reset the environment and get the initial state (position, velocity)
initial_state, info = env.reset(seed=42)
position = initial_state[0]
velocity = initial_state[1]
print(f"Car position: {position:.3f}, Velocity: {velocity:.3f}")

# Take random actions in the environment
for step in range(10):
    action = env.action_space.sample()  # Choose a random action: 0 (left), 1 (no push), 2 (right)
    state, reward, done, truncated, info = env.step(action)
    print(f"Step {step}: Position={state[0]:.3f}, Velocity={state[1]:.3f}, Reward={reward}")
    if done or truncated:
        print("Episode finished!")
        break

env.close()

import gymnasium as gym
from gymnasium.utils.play import play

play(
    gym.make('MountainCar-v0', render_mode='rgb_array'),
    keys_to_action={'j': 0, 'k': 2},  # j: move left, k: move right
    noop=1
)
