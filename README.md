# Adaptation of Multi-Agent Planning During Changes in the Environment

This repository contains preliminary results for research into scalable and autonomous multi-agent systems to address challenges in communication networks. The project uses **Multi-Agent Reinforcement Learning (MARL)** to explore decentralized, adaptive control mechanisms for networks, enabling advanced orchestration, security, and real-time adaptability. The experiments documented here serve as an early demonstration of MARL’s potential in dynamic environments.

## Preliminary Results

[![Watch the video](Adaptation.png)](https://www.youtube.com/watch?v=fmS5Ribn8Qo&ab_channel=AthmajanVivekananthan)


Initial experiments demonstrate the adaptability of MARL agents in a grid-world environment, where agents collaborate to intercept targets with variable behaviors. These findings lay the groundwork for applying MARL to real-world network challenges, such as dynamic resource allocation and network slicing.

- **[Basic Results on W&B](https://wandb.ai/athmajan-university-of-oulu/SecurityAndSurveillance/reports/Multiagent-Reinforcement-Learning-Rollout-and-Policy-Iteration--Vmlldzo4ODYxNDMx?accessToken=myfjbwjdmpdno7dz0ya9s4ty4f58ik9im0sqv3ki0i640qkhet8e818gffb6rw9m)**: Detailed initial results and analyses are available via this link.

## Future Work

Building on these preliminary findings, future work will focus on scaling these techniques to handle more complex, realistic environments, integrating continuous action spaces, and optimizing agent communication protocols.


### Steps to Run Experiments

#### 1. **Run Base Policy**
Use the following script to execute the base policy:
```bash
python Uncertainty_X/runRuleBasedAgent.py
```

#### 2. **Run Sequential Rollout**
Execute the script for a sequential rollout:
```bash
python Uncertainty_X/runSeqRollout.py
```

#### 3. **Run Standard Multi-Agent Rollout**
To run a standard multi-agent rollout:
```bash
python Uncertainty_X/runStandRollout.py
```

#### 4. **Learn to Model Others**
These scripts focus on learning to model the behavior of other agents:
```bash
python Uncertainty_X/learnRolloutOffV2.py
python Uncertainty_X/learnRollout_idqn.py
python Uncertainty_X/learn_idqn_CE.py
python Uncertainty_X/learn_idqn_L1.py
python Uncertainty_X/learn_idqn_mmse.py
python Uncertainty_X/classify_rmsProp.py
python Uncertainty_X/classify_kfold.py
```

#### 5. **Run Autonomous Multi-Agent Rollout**
To execute the autonomous multi-agent rollout script:
```bash
python Uncertainty_X/runAutoOffline.py
```

#### 6. **Run Cross Settings in Autonomous Multi-Agent Rollout**
This script handles cross-setting scenarios for autonomous multi-agent rollouts:
```bash
python Uncertainty_X/runApproxCross_30.py
```
