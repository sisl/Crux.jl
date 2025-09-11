# Examples

For a full set of examples, please see the [`examples/`](https://github.com/sisl/Crux.jl/tree/master/examples) directory.

- **Reinforcement learning examples**:
    - [Atari](https://github.com/sisl/Crux.jl/blob/master/examples/rl/atari.jl)
    - [Cart Pole](https://github.com/sisl/Crux.jl/blob/master/examples/rl/cartpole.jl)
    - [Half Cheetah (MuJoCo)](https://github.com/sisl/Crux.jl/blob/master/examples/rl/half_cheetah_mujoco.jl)
    - [Half Cheetah (PyBullet)](https://github.com/sisl/Crux.jl/blob/master/examples/rl/half_cheetah_pybullet.jl)
    - [Pendulum](https://github.com/sisl/Crux.jl/blob/master/examples/rl/pendulum.jl)
- **Imitation learning examples**:
    - [Cart Pole](https://github.com/sisl/Crux.jl/blob/master/examples/il/cartpole.jl)
    - [Half Cheetah (MuJoCo)](https://github.com/sisl/Crux.jl/blob/master/examples/il/half_cheetah_mujoco.jl)
    - [Lava World](https://github.com/sisl/Crux.jl/blob/master/examples/il/lavaworld.jl)
    - [Pendulum](https://github.com/sisl/Crux.jl/blob/master/examples/il/pendulum.jl)
- **Adversarial RL examples**:
    - [Cart Pole](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/cartpole.jl)
    - [Aircraft Collision Avoidance](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/collision_avoidance.jl)
    - [Continuous Bandit](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/continuous_bandit.jl)
    - [Continuous Pendulum](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/continuous_pendulum.jl)
    - [Discrete Pendulum](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/discrete_pendulum.jl)
    - [Continuum World](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/continuumworld.jl)
    - [Safety Gym](https://github.com/sisl/Crux.jl/blob/master/examples/adversarial/safety_gym.jl)
- **Offline RL examples**:
    - [Hopper Medium (MuJoCo)](https://github.com/sisl/Crux.jl/blob/master/examples/offline%20rl/hopper_medium.jl)


## Minimal RL Example

As a minimal example, we'll show how to set up a cart-pole problem and solve it with a simple Flux network using the REINFORCE algorithm.


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

You can run other algorithms, such as A2C and PPO, to generate different policies:
```julia
# Set up the critic network for actor-critic algorithms
V() = ContinuousNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 1)))

solver_a2c = A2C(S=S, π=ActorCritic(A(), V()))
policy_a2c = solve(solver_a2c, mdp)

solver_ppo = PPO(S=S, π=ActorCritic(A(), V()))
policy_ppo = solve(solver_ppo, mdp)
```

You also may want to adjust the number of environment interactions `N` or the number of interactions between updates `ΔN`:

```julia
solver_reinforce = REINFORCE(S=S, π=A(), N=10_000, ΔN=500)
policy_reinforce = solve(solver_reinforce, mdp)

solver_a2c = A2C(S=S, π=ActorCritic(A(), V()), N=10_000, ΔN=500)
policy_a2c = solve(solver_a2c, mdp)

solver_ppo = PPO(S=S, π=ActorCritic(A(), V()), N=10_000, ΔN=500)
policy_ppo = solve(solver_ppo, mdp)
```

### Plotting and Animations

You can take the above results and plot the learning curves:
```julia
p = plot_learning([solver_reinforce, solver_a2c, solver_ppo],
                  title="CartPole Training Curves",
                  labels=["REINFORCE", "A2C", "PPO"])
Crux.savefig(p, "cartpole_training.pdf")
```

Here's an example for the half cheetah MuJoCo problem, comparing four RL algorithms from [`examples/rl/half_cheetah_mujoco.jl`](https://github.com/sisl/Crux.jl/blob/master/examples/rl/half_cheetah_mujoco.jl#L78).

![mujoco](half_cheetah_mujoco_benchmark.pdf)


You can also create an animated gif of the final policy:
```julia
gif(mdp, policy_ppo, "cartpole_policy.gif", max_steps=100)
```

_Note: You may need to install `pygame` via `pip install "gymnasium[classic-control]"`_

