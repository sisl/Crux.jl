# Crux.jl

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:
### Reinforcement Learning
* <a href="./src/model_free/rl/dqn.jl">Deep Q-Learning</a>
  * Prioritized Experience Replay
* <a href="./src/model_free/rl/reinforce.jl">REINFORCE</a>
* <a href="./src/model_free/rl/ppo.jl">Proximal Policy Optimization (PPO)</a>
* <a href="./src/model_free/rl/a2c.jl">Advantage Actor Critic</a>
* <a href="./src/model_free/rl/ddpg.jl">Deep Deterministic Policy Gradient (DDPG)</a>
* <a href="./src/model_free/rl/td3.jl">Twin Delayed DDPG (TD3)</a>
* <a href="./src/model_free/rl/sac.jl">Soft Actor Critic (SAC)</a>

### Imitation Learning
* <a href="./src/model_free/il/bc.jl"> Behavioral Cloning </a>
* <a href="./src/model_free/il/gail.jl">Generative Adversarial Imitation Learning (GAIL)</a>
* <a href="./src/model_free/il/AdVIL.jl">Adversarial value moment imitation learning (AdVIL)</a>
* <a href="./src/model_free/il/AdRIL.jl">(AdRIL)</a>
* <a href="./src/model_free/il/sqil.jl">(SQIL)</a>
* <a href="./src/model_free/il/valueDICE.jl">valueDICE</a>
* <a href="./src/model_free/il/asaf.jl">Adversarial Soft Advantage Fitting (ASAF)</a>


Note for imitation learning algorithms: The parameter `normalize_demo::Bool` determines if the provided expert demonstrations should be normalized by the state and action spaces provided to the solver.

## Installation

* Install <a href="https://github.com/ancorso/POMDPGym">POMDPGym</a>
* Install by opening julia and running `]add git@github.com:ancorso/Crux.git`


To edit or contribute use `]dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

Maintained by Anthony Corso (acorso@stanford.edu)