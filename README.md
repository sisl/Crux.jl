# Crux.jl

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:

* <a href="./src/model_free/rl/dqn.jl">Deep Q-Learning</a>
  * Prioritized Experience Replay
* <a href="./src/model_free/rl/reinforce.jl">REINFORCE</a>
* <a href="./src/model_free/rl/ppo.jl">Proximal Policy Optimization (PPO)</a>
* <a href="./src/model_free/rl/a2c.jl">Advantage Actor Critic</a>
* <a href="./src/model_free/rl/ddpg.jl">Deep Deterministic Policy Gradient (DDPG)</a>
* <a href="./src/model_free/rl/td3.jl">Twin Delayed DDPG (TD3)</a>
* <a href="./src/model_free/rl/sac.jl">Soft Actor Critic (SAC)</a>
* <a href="./src/model_free/il/gail.jl">Generative Adversarial Imitation Learning (GAIL)</a>

## Installation

* Install <a href="https://github.com/ancorso/POMDPGym">POMDPGym</a>
* Install by opening julia and running `]add git@github.com:ancorso/Crux.git`


To edit or contribute use `]dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

Maintained by Anthony Corso (acorso@stanford.edu)