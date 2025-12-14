# Crux.jl

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://sisl.github.io/Crux.jl/dev/)
[![Build Status](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/Crux.jl/actions/workflows/CI.yml)
[![Code Coverage](https://codecov.io/gh/sisl/Crux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/Crux.jl)

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements deep reinforcement learning, imitation learning, batch RL, adversarial RL, and continual learning algorithms. See the [documentation](https://sisl.github.io/Crux.jl/dev/) for more details.

### Reinforcement Learning
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

### Imitation Learning
* [Behavioral Cloning](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/bc.jl)
* [Generative Adversarial Imitation Learning (GAIL) w/ On-Policy and Off-Policy Versions](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/gail.jl)
* [Adversarial Value Moment Imitation Learning (AdVIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/AdVIL.jl)
* [Adversarial Reward-moment Imitation Learning (AdRIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/AdRIL.jl)
* [Soft Q Imitation Learning (SQIL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/sqil.jl)
* [Adversarial Soft Advantage Fitting (ASAF)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/asaf.jl)
* [Inverse Q-Learning (IQLearn)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/il/iqlearn.jl)

### Batch RL
* [Batch Soft Actor Critic (BatchSAC)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/batch/sac.jl)
* [Conservative Q-Learning (CQL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/batch/cql.jl)

### Adversarial RL
* [Robust Adversarial RL (RARL)](https://github.com/sisl/Crux.jl/blob/master/src/model_free/adversarial/rarl.jl)

### Continual Learning
* [Experience Replay](https://github.com/sisl/Crux.jl/blob/master/src/model_free/cl/experience_replay.jl)


## Usage

An example usage of the `REINFORCE` algorithm with a simple Flux network for the Cart Pole problem is shown here:

```julia
using Crux, POMDPGym

# Problem setup
mdp = GymPOMDP(:CartPole)
as = actions(mdp)
S = state_space(mdp)

# Flux network: Map states to actions
A() = DiscreteNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, length(as))), as)

# Setup REINFORCE solver
solver_reinforce = REINFORCE(S=S, π=A())

# Solve the `mdp` to get the `policy`
policy_reinforce = solve(solver_reinforce, mdp)
```


## Installation

To install the package, run:
```
] add Crux
```

See the [installation documentation](https://github.com/ancorso/POMDPGym.jl?tab=readme-ov-file#installation) for more details on how to install POMDPGym for more environment.
