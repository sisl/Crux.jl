# Crux.jl

[![Build Status](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/sisl/Crux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/Crux.jl)

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:
### Reinforcement Learning
* <a href="./src/model_free/rl/dqn.jl">Deep Q-Learning</a>
  * Prioritized Experience Replay
* <a href="./src/model_free/rl/softq.jl">Soft Q-Learning</a>
* <a href="./src/model_free/rl/reinforce.jl">REINFORCE</a>
* <a href="./src/model_free/rl/ppo.jl">Proximal Policy Optimization (PPO)</a>
* Lagrange-Constrained PPO
* <a href="./src/model_free/rl/a2c.jl">Advantage Actor Critic</a>
* <a href="./src/model_free/rl/ddpg.jl">Deep Deterministic Policy Gradient (DDPG)</a>
* <a href="./src/model_free/rl/td3.jl">Twin Delayed DDPG (TD3)</a>
* <a href="./src/model_free/rl/sac.jl">Soft Actor Critic (SAC)</a>

### Imitation Learning
* <a href="./src/model_free/il/bc.jl"> Behavioral Cloning </a>
* <a href="./src/model_free/il/gail.jl">Generative Adversarial Imitation Learning (GAIL) w/ On-Policy and Off Policy Versions</a>
* <a href="./src/model_free/il/AdVIL.jl">Adversarial value moment imitation learning (AdVIL)</a>
* <a href="./src/model_free/il/AdRIL.jl">(AdRIL)</a>
* <a href="./src/model_free/il/sqil.jl">(SQIL)</a>
* <a href="./src/model_free/il/asaf.jl">Adversarial Soft Advantage Fitting (ASAF)</a>
* <a href="./src/model_free/il/iqlearn.jl">Inverse Q-Learning (IQLearn)</a>

### Batch RL
* <a href="./src/model_free/batch/sac.jl">Batch Soft Actor Critic (SAC)</a>
* <a href="./src/model_free/batch/cql.jl">Conservative Q-Learning (CQL)</a>

### Adversarial RL
* <a href="./src/model_free/adversarial/rarl.jl">Robust Adversarial RL (RARL)</a>

### Continual Learning
* Experience Replay


## Installation

* Install by opening julia and running `] add Crux`

To edit or contribute use `] dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

### Usage with POMDPGym

* Install <a href="https://github.com/ancorso/POMDPGym">POMDPGym</a>
    * See `examples/` for use of POMDPGym environments.

---
Maintained by Anthony Corso (acorso@stanford.edu) and Robert Moss (mossr@stanford.edu)
