using Crux, Flux, POMDPGym, Random, POMDPs
using Test 

mdp = GridWorldMDP(size = (10,10), tprob = .7)
as = [actions(mdp)...]
S = state_space(mdp)

N = 1000
Qnet = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4))

## cpu
𝒮 = DQNSolver(π = DQNPolicy(deepcopy(Qnet), as), S = S, N=N, rng = MersenneTwister(0))
p = solve(𝒮, mdp)

## gpu
𝒮_gpu = DQNSolver(π = DQNPolicy(deepcopy(Qnet) |> gpu, as), S = S, N=N, rng = MersenneTwister(0))
p = solve(𝒮_gpu, mdp)

s = rand(2, 100)
V1 = value(𝒮.π, s)
V2 = value(𝒮_gpu.π, s)
@test all(V1 .≈ V2)

## cpu - prioritized
buffer = ExperienceBuffer(S, DiscreteSpace(4), 1000, prioritized = true)
𝒮_prio =  DQNSolver(π = DQNPolicy(deepcopy(Qnet), as), S = S, N=N, buffer = buffer)
p = solve(𝒮_prio, mdp)

