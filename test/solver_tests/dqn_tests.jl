using Crux, Flux, POMDPGym
using Test 

g = GridWorldMDP(size = (10,10), tprob = .7)

N = 1000
Q = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4))

## DQN
𝒮 = DQNSolver(π = DQNPolicy(Q = deepcopy(Q), actions = actions(g)), sdim = 2, N=N, rng = MersenneTwister(0))
p = solve(𝒮, g)

𝒮_gpu = DQNSolver(π = DQNPolicy(Q = deepcopy(Q) |> gpu, actions = actions(g)), sdim = 2, N=N, rng = MersenneTwister(0))
p = solve(𝒮_gpu, g)

𝒮_prio = DQNSolver(π = DQNPolicy(Q = deepcopy(Q), actions = actions(g)), sdim = 2, N=N, buffer = ExperienceBuffer(2, 4, 1000, prioritized = true))
p = solve(𝒮_prio, g)

