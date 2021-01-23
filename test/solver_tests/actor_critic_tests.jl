using Crux, Flux, POMDPGym, Random, POMDPs
using Test

## Discete Tests
mdp = GridWorldMDP(size = (10,10), tprob = .7)
as = [actions(mdp)...]
S = state_space(mdp)

Vnet = ContinuousNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
Anet = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
AC(rng = Random.GLOBAL_RNG) = ActorCritic(DiscreteNetwork(deepcopy(Anet), as, rng = rng), deepcopy(Vnet))
AC_GPU(rng = Random.GLOBAL_RNG) = ActorCritic(DiscreteNetwork(deepcopy(Anet) |> gpu, as, rng = rng), deepcopy(Vnet) |> gpu)

# CPU
rng = MersenneTwister(0)
π1 = AC(rng)
solve(PGSolver(π = π1, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = AC(), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = AC(), S = S, loss = ppo()), mdp) # PPO

# GPU
rng = MersenneTwister(0)
π2 = AC_GPU(rng)
solve(PGSolver(π = π2, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = AC_GPU(), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = AC_GPU(), S = S, loss = ppo()), mdp) # PPO

s = rand(2, 100)
@test all(value(π1, s) .≈ value(π2, s))

## Continuous Tests
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
μnet = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
Vnet = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))

GP(rng = Random.GLOBAL_RNG) = GaussianPolicy(deepcopy(μnet), zeros(Float32, 1), rng = rng)
GP_GPU(rng = Random.GLOBAL_RNG) = GaussianPolicy(deepcopy(μnet) |> gpu, zeros(Float32, 1), rng = rng)

rng = MersenneTwister(0)
π1 = GP(rng)
solve(PGSolver(π = π1, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(GP(), V()), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = ActorCritic(GP(), V()), S = S, loss = ppo()), mdp) # PPO

rng = MersenneTwister(0)
π2 = GP_GPU(rng)
solve(PGSolver(π = π2, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(GP_GPU(), V() |> gpu), S = S, loss = a2c(), epochs = 1), mdp) # A2C
solve(PGSolver(π = ActorCritic(GP_GPU(), V() |> gpu), S = S, loss = ppo()), mdp) # PPO


s = rand(2, 100)
a = rand(1, 100)
@test mean(logpdf(π1, s, a) .- logpdf(π2, s, a)) < 1e-5

