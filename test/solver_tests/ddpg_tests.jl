using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)

# Solve with DDPG
𝒮_ddpg = DDPGSolver(π=ActorCritic(A(), QSA()), S=S, N=1000)
π_ddpg = solve(𝒮_ddpg, mdp)



