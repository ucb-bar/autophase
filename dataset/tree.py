import numpy as np
import scipy.io
from scipy import stats
import math
import random
import matplotlib.pyplot as plt
from prepare_train import load_data
from sklearn.ensemble import RandomForestClassifier
from gym_hls.envs.hls_env import HLSEnv
from gym_hls.envs.hls_multi_env import HLSMultiEnv

all_clfs={}
datasets = ['train_chstone_pgm.pkl','train_chstone_act.pkl','train_rand.pkl']
num_passes = 45
for dataset in datasets:
    clf = {}
    train_data, y_data = load_data(dataset)
    for p in range(num_passes):
        clf[p] = {"cls":RandomForestClassifier(),"useful":False}
        clf[p]["cls"].fit(train_data[p],y_data[p])
        clf[p]["feature_importance"]=clf[p]["cls"].feature_importances_
        print("pass -- ",p, 'importance -- ',clf[p]["cls"].feature_importances_)
        if(np.sum(clf[p]["cls"].feature_importances_)>1e-12):
            clf[p]["useful"] = True
            print("useful pass", p)
    all_clfs[dataset]=clf

def get_best_pass(clf,obs):
    useful_passes = []
    weights = []
    max_weight = 0
    sampler = []
    sampler_passes = []
    for p in clf.keys():
        if clf[p]["useful"] == False:
            continue
        sampler.append(np.sum(clf[p]["feature_importance"]))
        sampler_passes.append(p)
        pred = clf[p]["cls"].predict(obs)
        if pred[0] == 1:
            useful_passes.append(p)
            weights.append(np.sum(clf[p]["feature_importance"]))
    try:
        #print(useful_passes)
        best_pass = np.random.choice(np.array(useful_passes),1, p=np.array(weights)/sum(weights))
        return best_pass
    except:
        best_pass = np.random.choice(np.array(sampler_passes),1, p=np.array(sampler)/sum(sampler))
        return best_pass

from gym_hls.envs.chstone_bm import get_chstone, get_others
bms = get_chstone(N=12)
#from gym_hls.envs.random_bm import get_random
#bms = get_random(N=10)

env_configs={}
for i, bm in enumerate(bms):
  pgm, path = bm
  env_configs['pgm'] = pgm
  #env_configs['pgm_dir'] = path
  env_configs['pgm_dir']=path
  env_configs['run_dir'] = 'run_'+str(i)
  env_configs['verbose'] = True
  prog = HLSEnv(env_configs)
  traj_len = 12
  obs = prog.reset()
  act_obs = [0]*45
  traj = []
  applied_passes = []
  for step in range(traj_len):
    obs = np.array(obs)
    best_pass1 = get_best_pass(all_clfs['train_rand.pkl'],obs.reshape(1,obs.shape[0]))[0]
    best_pass2 = get_best_pass(all_clfs['train_chstone_act.pkl'],np.array(act_obs).reshape(1,45))[0]
    best_pass3 = get_best_pass(all_clfs['train_chstone_pgm.pkl'],obs.reshape(1,obs.shape[0]))[0]
    print(best_pass1,best_pass2,best_pass3)
    obs1,reward1 = prog.reset(init = applied_passes+[best_pass1],get_rew = True) #reset not step
    obs2,reward2 = prog.reset(init = applied_passes+[best_pass2],get_rew = True)
    obs3,reward3 = prog.reset(init = applied_passes+[best_pass3],get_rew = True)
    if reward1>=reward2 and reward1>=reward3:
        obs = obs1
        best_pass=best_pass1
    if reward3>=reward2 and reward3>=reward1:
        obs = obs3
        best_pass=best_pass3
    if reward2>=reward1 and reward2>=reward3:
        obs = obs2
        best_pass=best_pass2
    print(reward1,reward2,reward3)
    print(best_pass)
    act_obs[best_pass] += 1
    applied_passes.append(best_pass)

