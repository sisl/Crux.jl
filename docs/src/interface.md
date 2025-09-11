# Library/Interface

This section details the interface and functions provided by Crux.jl.

```@meta
CurrentModule = Crux
```

## Contents
```@contents
Pages = ["interface.md"]
```

## Index
```@index
Pages = ["interface.md"]
```

## Reinforcement Learning
RL interfaces are categorized by on-policy and off-policy below.

### On-Policy
```@docs
OnPolicySolver
a2c_loss
A2C
ppo_loss
PPO
lagrange_ppo_loss
LagrangePPO
reinforce_loss
REINFORCE
```

### Off-Policy
```@docs
OffPolicySolver
ddpg_target
smoothed_ddpg_target
ddpg_actor_loss
DDPG
dqn_target
DQN
sac_target
sac_deterministic_target
sac_max_q_target
sac_actor_loss
sac_temp_loss
SAC
softq_target
SoftQ
td3_target
td3_actor_loss
TD3
```

## Imitation Learning
```@docs
AdRIL
AdVIL
ASAF
BC
OnlineIQLearn
OffPolicyGAIL
OnPolicyGAIL
SQIL
```

## Adversarial RL
```@docs
AdversarialOffPolicySolver
RARL_DQN
RARL_TD3
RARL
```

## Continual Learning
```@docs
ExperienceReplay
```

## Batched RL
```@docs
BatchSolver
CQL
BatchSAC
```

## Policies
```@docs
PolicyParams
```
