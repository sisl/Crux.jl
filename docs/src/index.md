# Crux

[![Build Status](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/sisl/Crux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/Crux.jl)

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:

## Reinforcement Learning
* [Deep Q-Learning (DQN)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/dqn.jl)
  * Prioritized Experience Replay
* [Soft Q-Learning](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/softq.jl)
* [REINFORCE](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/reinforce.jl)
* [Proximal Policy Optimization (PPO)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/ppo.jl)
* [Lagrange-Constrained PPO](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/ppo.jl)
* [Advantage Actor Critic (A2C)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/a2c.jl)
* [Deep Deterministic Policy Gradient (DDPG)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/ddpg.jl)
* [Twin Delayed DDPG (TD3)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/td3.jl)
* [Soft Actor Critic (SAC)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/rl/sac.jl)

## Imitation Learning
* [Behavioral Cloning](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/bc.jl)
* [Generative Adversarial Imitation Learning (GAIL) w/ On-Policy and Off-Policy Versions](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/gail.jl)
* [Adversarial Value Moment Imitation Learning (AdVIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/AdVIL.jl)
* [Adversarial Reward-moment Imitation Learning (AdRIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/AdRIL.jl)
* [Soft Q Imitation Learning (SQIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/sqil.jl)
* [Adversarial Soft Advantage Fitting (ASAF)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/asaf.jl)
* [Inverse Q-Learning (IQLearn)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/iqlearn.jl)

## Batch RL
* [Batch Soft Actor Critic (BatchSAC)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/batch/sac.jl)
* [Conservative Q-Learning (CQL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/batch/cql.jl)

## Adversarial RL
* [Robust Adversarial RL (RARL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/adversarial/rarl.jl)

## Continual Learning
* [Experience Replay](https://github.com/sisl/Crux.jl/blob/master/src/model_free/cl/experience_replay.jl)

---

## Citation

```
In progress.
```