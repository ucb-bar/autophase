import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

ray.init()
env_configs = {}

num_pgms = 1
from gym_hls.envs.chstone_bm import get_chstone, get_others, get_all9
bms = get_all9()
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs['pgm'] = pgm
  env_configs['pgm_dir'] = path
  env_configs['run_dir'] = 'run_'+str(i)
  env_configs['feature_type'] = 'act_hist'
  env_configs['verbose'] = True

  tune.run_experiments({
      "my_experiment": {
          "run": "DQN",
          "env":HLSEnv,
          "checkpoint_freq": 2,
          "stop": {"episodes_total": 100},
          "config": {

      # === Model ===
      "horizon" : 12,
      "n_step": 3,
      # Number of atoms for representing the distribution of return. When
      # this is greater than 1, distributional Q-learning is used.
      # the discrete supports are bounded by v_min and v_max
      "num_atoms": 1,
      "v_min": -10.0,
      "v_max": 10.0,
      # Whether to use noisy network
      "noisy": False,
      # control the initial value of noisy nets
      "sigma0": 0.5,
      # Whether to use dueling dqn
      "dueling": True,
      # Whether to use double dqn
      "double_q": True,
      # Hidden layer sizes of the state and action value networks
      "hiddens": [256],
      # N-step Q learning
     # === Evaluation ===
      # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
      # The evaluation stats will be reported under the "evaluation" metric key.
      # Note that evaluation is currently not parallelized, and that for Ape-X
      # metrics are already only reported for the lowest epsilon workers.
      #"evaluation_interval": None,
      # Number of episodes to run per evaluation period.
      #"evaluation_num_episodes": 10,

      # === Exploration ===
      # Max num timesteps for annealing schedules. Exploration is annealed from
      # 1.0 to exploration_fraction over this number of timesteps scaled by
      # exploration_fraction
      "schedule_max_timesteps": 1000,
      # Number of env steps to optimize for before returning
      "timesteps_per_iteration": 4,
      # Fraction of entire training period over which the exploration rate is
      # annealed
      "exploration_fraction": 0.1,
      # Final value of random action probability
      "exploration_final_eps": 0.02,
      # Update the target network every `target_network_update_freq` steps.
      "target_network_update_freq": 10,
      # Use softmax for sampling actions.
      #"soft_q": False,
      # Softmax temperature. Q values are divided by this value prior to softmax.
      # Softmax approaches argmax as the temperature drops to zero.
      #"softmax_temp": 1.0,
      # If True parameter space noise will be used for exploration
      # See https://blog.openai.com/better-exploration-with-parameter-noise/
      #"parameter_noise": False,

      # === Replay buffer ===
      # Size of the replay buffer. Note that if async_updates is set, then
      # each worker will have a replay buffer of this size.
      "buffer_size": 2000,
      # If True prioritized replay buffer will be used.
      "prioritized_replay": True,
      # Alpha parameter for prioritized replay buffer.
      "prioritized_replay_alpha": 0.6,
      # Beta parameter for sampling from prioritized replay buffer.
      "prioritized_replay_beta": 0.4,
      # Fraction of entire training period over which the beta parameter is
      # annealed
      "beta_annealing_fraction": 0.2,
      # Final value of beta
      "final_prioritized_replay_beta": 0.4,
      # Epsilon to add to the TD errors when updating priorities.
      "prioritized_replay_eps": 1e-6,
      # Whether to LZ4 compress observations
      "compress_observations": True,

      # === Optimization ===
      # Learning rate for adam optimizer
      "lr": 5e-4,
      # Adam epsilon hyper parameter
      "adam_epsilon": 1e-8,
      # If not None, clip gradients during optimization at this value
      "grad_norm_clipping": 40,
      # How many steps of the model to sample before learning starts.
      "learning_starts": 1000,
      # Update the replay buffer with this many samples at once. Note that
      # this setting applies per-worker if num_workers > 1.
      "sample_batch_size": 4,
      # Size of a batched sampled from replay buffer for training. Note that
      # if async_updates is set, then each worker returns gradients for a
      # batch of this size.
      "train_batch_size": 32,
      "env_config": env_configs,
          },
      },
  })
