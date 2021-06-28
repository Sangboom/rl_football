from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn

num_agents = 11
num_policies = 11

class RllibGFootball(MultiAgentEnv):
  """An example of a wrapper for GFootball to make it compatible with rllib."""

  def __init__(self, num_agents):
    self.env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic', stacked=False,  # academy_single_goal_versus_lazy, academy_counterattack_hard, 11_vs_11_easy_stochastic
        logdir='/tmp/rllib_test',
        write_goal_dumps=False, write_full_episode_dumps=False, render=False,
        dump_frequency=0,
        number_of_left_players_agent_controls=num_agents,
        channel_dimensions=(42, 42))
    self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)
    self.num_agents = num_agents

  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(actions)
    rewards = {}
    obs = {}
    infos = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      infos[key] = i
      if self.num_agents > 1:
        rewards[key] = r[pos]
        obs[key] = o[pos]
      else:
        rewards[key] = r
        obs[key] = o
    dones = {'__all__': d}
    return obs, rewards, dones, infos

# model default 찾고 삐리하면 custom model 만들기 
"""
class my_model(TorchModelV2, nn.Module):

  def __init__(self, *args, **kwargs):
    TorchModelV2.__init__(self, *args, **kwargs)
    nn.Module.__init__(self)
    self._hidden_layers = nn.Sequential(
      nn.Linear(obs_size, 256),
      nn.Tanh(),
      nn.Linear(256, 256),
      nn.Tanh(),
      nn.Linear(256, 19),
      softmax
      )
"""

if __name__ == '__main__':
  ray.init(num_gpus=1)

  # Simple environment with `num_agents` independent players
  register_env('gfootball', lambda _: RllibGFootball(num_agents))
  single_env = RllibGFootball(num_agents)
  obs_space = single_env.observation_space
  act_space = single_env.action_space

  def gen_policy(_):
    return (None, obs_space, act_space, {})

  # Setup PPO with an ensemble of `num_policies` different policies
  policies = {
      'policy_{}'.format(i): gen_policy(i) for i in range(num_policies)
  }
  policy_ids = list(policies.keys())

  tune.run(
      'PPO',
      stop={'training_iteration': 100000},
      checkpoint_freq=50,
      restore="/home/sangbeom/ray_results/PPO/PPO_gfootball_40f31_00000_0_2020-12-07_14-51-30/checkpoint_6000/checkpoint-6000",
      config={
          'framework': 'torch',
          'env': 'gfootball',
          'lambda': 0.95,
          'kl_coeff': 0.2,
          'clip_rewards': False,
          'vf_clip_param': 10.0,
          'entropy_coeff': 0.01,
          'train_batch_size': 2000,
#          'sample_batch_size': 100,
          'sgd_minibatch_size': 500,
          'num_sgd_iter': 10,
          'num_workers': 3,
          'num_envs_per_worker': 1,
          'batch_mode': 'truncate_episodes',
          'observation_filter': 'NoFilter',
          'vf_share_layers': 'true',
          'monitor': True,
          'num_gpus': 1,
          'lr': 2.5e-4,
          'log_level': 'DEBUG',
          'simple_optimizer': True,
          'multiagent': {
              'policies': policies,
              'policy_mapping_fn': tune.function(
                  lambda agent_id: policy_ids[int(agent_id[6:])]),
          },
      },
  )

