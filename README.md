# AutoPhase

## Overview 
AutoPhase is a framework that takes a program and ﬁnds a sequence of passes that optimize the performance of the generated circuit. 
AutoPhase is a framework that uses deep reinforcement learning (RL) to optimize the order of the passes that increase the performance of the RTL by minimizing the clock cycle count. The framework uses LLVM at the frontend and LegUp backend tools. More details are available on the paper. 
Dependencies:
- Ray[rllib] (`pip3 install ray[rllib]`) 
- Tensorflow (https://www.tensorflow.org/install/pip)
- LegUp (http://legup.eecg.utoronto.ca/docs/4.0/gettingstarted.html#getstarted)
  - LLVM (comes with LegUp)
  - Clang (comes with LegUp)

This framework takes the input of a program compiles into LLVM IR. The neural network agent takes as input the features of the program(using IR Feature Extractor), clock cycle count(using Clock-cycle Profiler), and histogram of previously applied passes. Then it outputs the prediction of the best optimization passes to apply, which is used to generate new LLVM IR.  

## Running training on AutoPhase

```
- git clone https://github.com/hqjenny/AutoPhase
- cd AutoPhase/gym-hls/
- pip install -e .
- cd ../algos/rl
- python  ppo_ray.py
```
