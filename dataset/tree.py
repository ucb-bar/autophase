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
from gym_hls.envs.getcycle import getPasses
features = ["# of BB where total args for phi nodes > 5", "# of BB where total args for phi nodes is [1, 5]", "# of BB's with 1 predecessor", "# of BB's with 1 predecessor and 1 successor", "# of BB's with 1 predecessor and 2 successors", "# of BB's with 1 successor", "# of BB's with 2 predecessors", "# of BB's with 2 predecessors and 1 successor", "# of BB's with 2 predecessors and successors", "# of BB's with 2 successors", "# of BB's with >2 predecessors", "# of BB's with Phi node # in range (0, 3]", "# of BB's with more than 3 Phi nodes", "# of BB's with no Phi nodes", "# of Phi-nodes at beginning of BB", "# of branches", "# of calls that return an int", "# of critical edges", "# of edges", "# of occurrences of 32-bit integer constants", "# of occurrences of 64-bit integer constants", "# of occurrences of constant 0", "# of occurrences of constant 1", "# of unconditional branches", "Binary operations with a constant operand", "Number of AShr insts", "Number of Add insts", "Number of Alloca insts", "Number of And insts", "Number of BB's with instructions between [15, 500]", "Number of BB's with less than 15 instructions", "Number of BitCast insts", "Number of Br insts", "Number of Call insts", "Number of GetElementPtr insts", "Number of ICmp insts", "Number of LShr insts", "Number of Load insts", "Number of Mul insts", "Number of Or insts", "Number of PHI insts", "Number of Ret insts", "Number of SExt insts", "Number of Select insts", "Number of Shl insts", "Number of Store insts", "Number of Sub insts", "Number of Trunc insts", "Number of Xor insts", "Number of ZExt insts", "Number of basic blocks", "Number of instructions (of all types)", "Number of memory instructions", "Number of non-external functions", "Total arguments to Phi nodes", "Unary"]
opt_passes_str = "-correlated-propagation -scalarrepl -lowerinvoke -strip -strip-nondebug -sccp -globalopt -gvn -jump-threading -globaldce -loop-unswitch -scalarrepl-ssa -loop-reduce -break-crit-edges -loop-deletion -reassociate -lcssa -codegenprepare -memcpyopt -functionattrs -loop-idiom -lowerswitch -constmerge -loop-rotate -partial-inliner -inline -early-cse -indvars -adce -loop-simplify -instcombine -simplifycfg -dse -loop-unroll -lower-expect -tailcallelim -licm -sink -mem2reg -prune-eh -functionattrs -ipsccp -deadargelim -sroa -loweratomic -terminate".split()


all_clfs={}
#datasets = ['train_chstone_pgm.pkl','train_chstone_act.pkl','train_rand.pkl']
num_passes = 45
datasets = ['train_rand.pkl']
important_passes=[]
for dataset in datasets:
    clf = {}
    importances = []
    train_data, y_data = load_data(dataset)
    for p in range(num_passes):
        clf[p] = {"cls":RandomForestClassifier(criterion='entropy',random_state=11),"useful":False}
        clf[p]["cls"].fit(train_data[p],y_data[p])
        clf[p]["feature_importance"]=clf[p]["cls"].feature_importances_
        importances.append(clf[p]["feature_importance"])
        print("pass -- ",p, 'importance -- ',clf[p]["cls"].feature_importances_)
        if(np.sum(clf[p]["cls"].feature_importances_)>1e-12):
            clf[p]["useful"] = True
            print("useful pass", p)
            important_passes.append(p)
    print(np.unique(np.array(important_passes)))
    all_clfs[dataset]=clf

    npimp = np.array(importances)
    bad_features = []
    bad_passes = []
    good_features=[]
    good_passes=[]
    for i in range(56):
        if np.max(npimp[:,i])<0.05:
            bad_features.append(i)
        else:
            good_features.append(i)
    good_couples=[]
    for i in range(45):
        if np.max(npimp[i,:])<0.05:
            bad_passes.append(i)
        else:
            good_passes.append(i)
        for j in range(56):
            if npimp[i,j] > 0.1:
                good_couples.append([i,j,opt_passes_str[i],features[j]])
    print(dataset)
    print('bad_features',bad_features)
    print('bad_passes',bad_passes)
    print('good_features',good_features)
    print('good_passes',good_passes)
    print('good pass, feature couples', good_couples)
    fig1=plt.figure(1,figsize=(9,9))
    ax1=fig1.add_subplot(1,1,1)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    imtplot1=plt.imshow(importances)
    #plt.rc('font', size=14)
    #plt.rc('axes',titlesize=12)
    #plt.rc('axes',labelsize=12)
    #plt.rc('xtick', labelsize=1)
    #plt.rc('ytick',labelsize=1)
    plt.colorbar()
    plt.xlabel('feature')
    plt.ylabel('pass')
    plt.title('importance',y=1.14)
    plt.show()


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
  traj_len = 45
  obs = prog.reset()
  act_obs = [0]*45
  traj = []
  applied_passes = []
  for step in range(traj_len):
    obs = np.array(obs)
    best_pass1 = get_best_pass(all_clfs['train_rand.pkl'],obs.reshape(1,obs.shape[0]))[0]
    #best_pass2 = get_best_pass(all_clfs['train_chstone_act.pkl'],np.array(act_obs).reshape(1,45))[0]
    best_pass3 = get_best_pass(all_clfs['train_chstone_pgm.pkl'],obs.reshape(1,obs.shape[0]))[0]
    #print(best_pass1,best_pass2,best_pass3)
    print(best_pass1,best_pass3)
    obs1,reward1 = prog.reset(init = applied_passes+[best_pass1],get_rew = True) #reset not step
    #obs2,reward2 = prog.reset(init = applied_passes+[best_pass2],get_rew = True)
    obs3,reward3 = prog.reset(init = applied_passes+[best_pass3],get_rew = True)
    '''
    if reward1>=reward2 and reward1>=reward3:
        obs = obs1
        best_pass=best_pass1
    if reward3>=reward2 and reward3>=reward1:
        obs = obs3
        best_pass=best_pass3
    if reward2>=reward1 and reward2>=reward3:
        obs = obs2
        best_pass=best_pass2'''

    if(reward1>=reward3):
        obs=obs3
        best_pass=best_pass3
    else:
        obs=obs1
        best_pass=best_pass1

    #print(reward1,reward2,reward3)
    print(reward1,reward3)
    print(best_pass)
    act_obs[best_pass] += 1
    applied_passes.append(best_pass)

