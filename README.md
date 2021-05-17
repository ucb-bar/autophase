# AutoPhase: Compiler Phase-Ordering for HLS with Deep Reinforcement Learning

AutoPhase is a framework that uses deep reinforcement learning (RL) to optimize the order of the passes that increase the performance of the programs by minimizing the clock cycle count. For more details, please see: 
- [MLSys'20 AutoPhase Paper](https://arxiv.org/pdf/2003.00671.pdf)
- [MLSys'19 AutoPhase Presentation](https://people.eecs.berkeley.edu/~qijing.huang/2020MLSys/2020SysML_AutoPhase_Presentation.pdf)
- [FCCM'19 AutoPhase Paper](https://arxiv.org/pdf/1901.04615.pdf)

This framework takes the input of a program compiles into LLVM IR. The neural network agent takes as input the features of the program (using IR Feature Extractor), clock cycle count (using Clock-cycle Profiler), and histogram of previously applied passes. Then it outputs the prediction of the best optimization passes to apply, which is used to generate new LLVM IR.  

## Installation
AutoPhase depends on the following open-source tools, LegUp (HLS compiler), and Ray (RL framework).
Dependencies:
- Ray[rllib] (`pip3 install ray[rllib]`) 
- Tensorflow (https://www.tensorflow.org/install/pip)
- LegUp4.0 (http://legup.eecg.utoronto.ca/docs/4.0/gettingstarted.html#getstarted)
  - LLVM (comes with LegUp)
  - Clang (comes with LegUp)


Please refer to (patch/README.md) to install the LLVM patches. 

We also compare RL against:
- insertion-based greedy search / beam search (algos/greedy/greedy\_search.py)
- DEAP (`pip install deap`)
- OpenTuner (`pip install opentuner`)

## Run AutoPhase
```
git clone https://github.com/ucb-bar/autophase.git 
export LEGUP_PATH=$(realpath "/path/to/legup-4.0")
export AUTOPHASE_PATH=$(realpath "/path/to/autophase")
cd autophase/gym-hls/
pip install -e .
cd ../algos/rl
python ppo_ray.py
```
