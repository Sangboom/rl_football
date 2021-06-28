import gfootball.env as football_env
from gfootball.env.wrappers import MultiAgentToSingleAgent

env = football_env.create_environment(
    env_name="11_vs_11_stochastic", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, 
    render=True, number_of_left_players_agent_controls=11, representation='simple115', rewards='scoring, checkpoint')
env.reset()
steps = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  single_obs = MultiAgentToSingleAgent.get_observation(obs)

  print(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    print("Step %d Reward: %f" % (steps, rew))
  if done:
    break

print("Steps: %d Reward: %.2f" % (steps, rew))