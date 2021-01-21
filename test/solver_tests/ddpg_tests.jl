using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
DDPG_Q() = Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1),x -> 200f0*x .- 200f0)
μ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))

# Solve with DDPG
𝒮_ddpg = DDPGSolver(π = DDPGPolicy(μ(), DDPG_Q(), action_dim = 1), S = S, N=1000)
solve(𝒮_ddpg, mdp)


