# Proximal Policy Optimization 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Classic control**

Acrobot-v1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;CartPole-v1&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;MountainCarContinuous-v0&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Pendulum-v0

<p>
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/Acrobot-v1.gif" width="25%" height="25%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/CartPole-v1.gif" width="25%" height="25%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/MountainCarContinuous-v0.gif" width="24%" height="24%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/Pendulum-v0.gif" width="24%" height="24%">
</p>

<p>
&emsp;
</p>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Box2D**&emsp;&emsp; &emsp; &emsp; &emsp;&emsp;&emsp; &emsp;&emsp; &emsp; &emsp; &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Roboschool**



LunarLanderContinuous-v2&emsp;&emsp;&emsp;BipedalWalker-v3&emsp;&emsp; &emsp;&emsp;RoboschoolAnt-v1&emsp;&emsp;&emsp;&emsp;&emsp;RoboschoolHalfCheetah-v1
<p>
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/LunarLanderContinuous-v2.gif" width="24%" height="20%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/BipedalWalker-v3.gif" width="24%" height="20%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/RoboschoolAnt-v1.gif" width="25%" height="35%">
  <img src="https://github.com/mandrakedrink/PPO-pytorch/blob/master/gifs/RoboschoolHalfCheetah-v1.gif" width="25%" height="35%">
</p>

### This repository contains the source code pytorch realization of PPO for solving openai gym enviroments.

Proximal Policy Optimization is the one of state of the art reinforcement learning algorithm, its main feature is the control of policy changes, we don't want to deviate too much from the old policy when recalculating weights.

All solved environments with hyperparameters are available here
[opeanai_ppo.ipynb](https://github.com/mandrakedrink/PPO-pytorch/blob/master/opeanai_ppo.ipynb)[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1ejncwwEshbPseOtXN3NzeF2lvMF0CTf_?usp=sharing)<br>
The code is based on this repository - https://github.com/MrSyee/pg-is-all-you-need.
