import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

def handler(signum, frame):
    print ('Signal handler called with signal', signum)
    exit()

signal.signal(signal.SIGALRM, handler)
signal.alarm(2*21600)

ray.init()
env_config = {
    'normalize': False,
    'orig_and_normalize':False,
    'log_obs_reward':True,
    #'verbose':True,
    'bm_name':'random',
    'num_pgms':100}

tune.run_experiments({
    "ppo_lstm": {
        "run": "PPO",
        "env":HLSMultiEnv,
        "checkpoint_freq": 10,
        "stop": {"episode_reward_mean": 1000000},
        "config": {
            "sample_batch_size": 50,
            "train_batch_size": 200,
            "sgd_minibatch_size": 40,
#            "model": {"use_lstm": True},
            "horizon": 12,
            "num_gpus": 2,
            "num_workers": 7,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_config,
        },
    },
})
