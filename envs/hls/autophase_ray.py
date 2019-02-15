import ray
import ray.tune as tune
from ray.rllib.agents import ppo
import env_autophase_ray
import multi_env

ray.init()
env_configs = {}
from chstone_bm import get_chstone
bm = get_chstone()
pgm,pgm_dir = sorted(bm)[0]
print(pgm,pgm_dir)
#pgm_dir="./examples"; 
run_dir=None; delete_run_dir=True; init_with_passes=True
env_configs['pgm'] =pgm
env_configs['pgm_dir'] = pgm_dir
env_configs['run_dir'] = run_dir
env_configs['delete_run_dir'] = delete_run_dir
env_configs['init_with_passes'] = init_with_passes
 
tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        #"env": env_autophase_ray.CompilationEnv,
        "env":multi_env.HLSMultiEnv,
        "stop": {"episode_reward_mean": 0},
        "config": {
            "sample_batch_size": 100,
            "train_batch_size": 700,
            "sgd_minibatch_size": 70,
            "horizon": 10,
            "num_gpus": 2,
            "num_workers": 2,
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "env_config": env_configs,  # config to pass to env class
        },
    },
})
