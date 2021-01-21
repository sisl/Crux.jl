using Crux, Flux, POMDPGym, Random, POMDPs
using Test

## Discete Tests
mdp = GridWorldMDP(size = (10,10), tprob = .7)
as = [actions(mdp)...]
S = state_space(mdp)

Vnet = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1))
Anet = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)

# CPU
rng = MersenneTwister(0)
π1 = CategoricalPolicy(deepcopy(Anet), as, rng = rng)
solve(PGSolver(π = π1, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(CategoricalPolicy(deepcopy(Anet),  as), deepcopy(Vnet)), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = ActorCritic(CategoricalPolicy(deepcopy(Anet), as), deepcopy(Vnet)), S = S, loss = ppo()), mdp) # PPO

# GPU
rng = MersenneTwister(0)
π2 = CategoricalPolicy(deepcopy(Anet) |> gpu, as, rng = rng)
solve(PGSolver(π = π2, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(CategoricalPolicy(deepcopy(Anet) |> gpu, as), deepcopy(Vnet) |> gpu), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = ActorCritic(CategoricalPolicy(deepcopy(Anet) |> gpu, as), deepcopy(Vnet) |> gpu), S = S, loss = ppo()), mdp) # PPO

s = rand(2, 100)
@test all(logits(π1, s) .≈ logits(π2, s))

## Continuous Tests
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
μ() = Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))
V() = Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))

μ1 = μ()
rng = MersenneTwister(0)
π1 = GaussianPolicy(deepcopy(μ1), zeros(Float32, 1), rng = rng)
solve(PGSolver(π = π1, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(GaussianPolicy(μ(), zeros(Float32, 1)), V()), S = S, loss = a2c()), mdp) # A2C
solve(PGSolver(π = ActorCritic(GaussianPolicy(μ(), zeros(Float32, 1)), V()), S = S, loss = ppo()), mdp) # PPO

rng = MersenneTwister(0)
π2 = GaussianPolicy(deepcopy(μ1) |> gpu, zeros(Float32, 1) |> gpu, rng = rng)
solve(PGSolver(π = π2, S = S, loss = reinforce(), rng = rng), mdp) # REINFORCE
solve(PGSolver(π = ActorCritic(GaussianPolicy(μ() |> gpu, zeros(Float32, 1) |> gpu), V() |> gpu), S = S, loss = a2c(), epochs = 1), mdp) # A2C
solve(PGSolver(π = ActorCritic(GaussianPolicy(μ() |> gpu, zeros(Float32, 1) |> gpu), V() |> gpu), S = S, loss = ppo()), mdp) # PPO


s = rand(2, 100)
a = rand(1, 100)
@test mean(logpdf(π1, s, a) .- logpdf(π2, s, a)) < 1e-5

