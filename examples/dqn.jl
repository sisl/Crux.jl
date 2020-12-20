using Crux, Flux
include("mdps/gridworld.jl")

g = SimpleGridWorld(size = (10,10), tprob = .7)

N = 10000
Q = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4))

## DQN
𝒮 = DQNSolver(π = DQNPolicy(deepcopy(Q), actions(g)), sdim = 2, N=N, rng = MersenneTwister(0))
@profiler p = solve(𝒮, g)

𝒮_gpu = DQNSolver(π = DQNPolicy(deepcopy(Q), actions(g), device = gpu), sdim = 2, N=N, rng = MersenneTwister(0))
@time p = solve(𝒮_gpu, g)

𝒮_prio= DQNSolver(π = DQNPolicy(deepcopy(Q), actions(g)), sdim = 2, N=N, buffer = ExperienceBuffer(2, 4, 1000, prioritized = true))
p = solve(𝒮_prio, g)

