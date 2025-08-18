import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. Create a vectorized environment
# make_vec_env allows running multiple environments in parallel for faster data collection
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# 2. Initialize the PPO model
# "MlpPolicy" refers to a Multi-Layer Perceptron policy (suitable for simple environments)
# vec_env is the environment to train on
# verbose=1 provides training progress output
model = PPO("MlpPolicy", vec_env, verbose=1)

# 3. Train the model
# total_timesteps defines the total number of interactions with the environment
model.learn(total_timesteps=250000)

# 4. (Optional) Save the trained model
model.save("ppo_cartpole")

# 5. (Optional) Load the trained model and test
del model # Remove the current model to demonstrate loading
model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if True in dones: # Break if any environment in the vectorized setup is done
        break

vec_env.close() # Close the environment window