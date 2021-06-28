import gfootball.env as football_env

import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers
from pfrl.agents import PPO


env = football_env.create_environment(env_name="academy_empty_goal_close", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=True)
env = pfrl.wrappers.CastObservationToFloat32(env)
env.reset()
steps = 0

obs_space = env.observation_space
action_space = env.action_space
print(obs_space)

obs_size = obs_space.low.size

# Normalize observations based on their empirical mean and variance
obs_normalizer = pfrl.nn.EmpiricalNormalization(
    obs_space.low.size, clip_threshold=5
)

policy = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 19),
    pfrl.policies.SoftmaxCategoricalHead(),
)

vf = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)
# While the original paper initialized weights by normal distribution,
# we use orthogonal initialization as the latest openai/baselines does.
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

ortho_init(policy[0], gain=1)
ortho_init(policy[2], gain=1)
ortho_init(policy[4], gain=1e-2)
ortho_init(vf[0], gain=1)
ortho_init(vf[2], gain=1)
ortho_init(vf[4], gain=1)

# Combine a policy and a value function into a single model
model = pfrl.nn.Branched(policy, vf)

opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

agent = PPO(
    model,
    opt,
    obs_normalizer=obs_normalizer,
    gpu=0,
    update_interval=2048,
    minibatch_size=64,
    epochs=10,
    clip_eps_vf=None,
    entropy_coef=0,
    standardize_advantages=True,
    gamma=0.995,
    lambd=0.97,
)

agent.load('FBPPO')

n_episodes = 10000
max_episode_len = 200
for i in range(1, n_episodes + 1):
    obs = env.reset()
    obs = np.reshape(obs, (27648))
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        #action = action[0]
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
        reset = t == max_episode_len
        obs = np.reshape(obs, (27648))
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')