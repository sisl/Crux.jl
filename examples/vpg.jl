using Shard, Flux
include("mdps/gridworld.jl")

mdp = SimpleGridWorld(size = (10,10), tprob = .7)

N = 200000
Q = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4), softmax)
V = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 32, relu), Dense(32, 32, relu), Dense(32, 1))

## VPG on CPU
rng = MersenneTwister(2)
𝒮 = VPGSolver(π = CategoricalPolicy(deepcopy(Q), actions(mdp), rng = rng), sdim = 2, adim = 4, N=N, batch_size = 128, baseline = Baseline(deepcopy(V)), rng = rng)
p = solve(𝒮, mdp)

## VPG on GPU
rng = MersenneTwister(2)
𝒮 = VPGSolver(π = CategoricalPolicy(deepcopy(Q), actions(mdp), rng = rng, device = gpu), sdim = 2, adim = 4, N=N, batch_size = 128, baseline = Baseline(deepcopy(V), device = gpu), rng = rng)
p = solve(𝒮, mdp)

