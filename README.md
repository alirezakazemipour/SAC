# SAC
Implementation of Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor    
>Added another branch for [Soft Actor-Critic Algorithms and Applications ](https://arxiv.org/pdf/1812.05905.pdf) -> [SAC_V1](https://github.com/alirezakazemipour/SAC/tree/SAV_V1).  

## Demo


## Dependencies
- gym == 0.17.2  
- mujoco-py == 2.0.2.13  
- numpy == 1.19.1  
- psutil == 5.4.2  
- torch == 1.4.0  
## Installation
```shell
pip3 install -r requirements.txt
```

## Environment tested
- [x] Humanoid-v2
- [x] Hopper-v2
- [x] Walker2d-v2 
- [ ] HalfCheetah-v2 

## Results


## Reference
1. [_Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor_, Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)
2. [_Soft Actor-Critic Algorithms and Applications_, Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905)

## Acknowledgement
All credits goes to [@pranz24](https://github.com/pranz24) for his brilliant Pytorch [implementation of SAC](https://github.com/pranz24/pytorch-soft-actor-critic).  
Special thanks to [@p-christ](https://github.com/p-christ) for [SAC.py](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/a8bd4f99f03b7d0a8e3dabd31fdc91490e506221/agents/actor_critic_agents/SAC.py)  