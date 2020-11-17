include("../src/includes.jl") # this will be replaced with a using statement eventually
include("mdps/gridworld.jl")

g = SimpleGridWorld(size = (10,10), tprob = .7)

N = 100000
Q(args...) = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4), args...)

## DQN
cpusolver = DQNSolver(π = DQNPolicy(Q(), g), N=N, batch_size = 128, exploration_policy = EpsGreedyPolicy(g, LinearDecaySchedule(start=1., stop=0.1, steps=N/2)))
p = solve(cpusolver, g)

gpusolver = DQNSolver(π = DQNPolicy(Q(), g, device = gpu), N=N, batch_size = 128, exploration_policy = EpsGreedyPolicy(g, LinearDecaySchedule(start=1., stop=0.1, steps=N/2)), device = gpu)
gpu_pol = solve(gpusolver, g)

## VPG
bline = Baseline(Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 1)))
π = CategoricalPolicy(Q(softmax), g)
𝒮 = VPGSolver(π = π, N = N, baseline = bline)
p = solve(𝒮, g)

bline = Baseline(Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 1)), device = gpu)
π = CategoricalPolicy(Q(softmax), g, device = gpu)
𝒮 = VPGSolver(π = π, N = N, baseline = bline, device = gpu)
p = solve(𝒮, g)

bline.V_GPU.layers[2].W

