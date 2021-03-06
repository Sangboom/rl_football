from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
tune.run(PPOTrainer, config={"env": "CartPole-v0", "framework": "torch", "monitor": True, "num_workers": 0})