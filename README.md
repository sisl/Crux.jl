# Crux.jl

Deep RL library with concise implementations of popular algorithms. Implemented using [Flux.jl](https://github.com/FluxML/Flux.jl) and fits into the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

Supports CPU and GPU computation and implements the following algorithms:

* <a href="./src/solvers/dqn.jl">Deep Q-Learning</a>
  * Prioritized Experience Replay
* <a href="./src/solvers/actor_critic.jl">Vanilla Policy Gradient</a>
  * REINFORCE
  * Proximal Policy Optimization (PPO)
  * Advantage Actor-Critic (A2C)
* <a href="./src/solvers/ddpg.jl">Deep Deterministic Policy Gradient (DDPG)</a>
* <a href="./src/solvers/gail.jl">Generative Adversarial Imitation Learning (DQN-GAIL)</a>

## Installation

Install by opening julia and running `]add git@github.com:ancorso/Crux.git`
To edit or contribute use `]dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

Maintained by Anthony Corso (acorso@stanford.edu)