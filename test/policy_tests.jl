using Crux
using Test
using Flux
using POMDPs
using POMDPGym
using POMDPPolicies
using Random
using Statistics
using Distributions


## DQN Policy
π_dqn = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)), actions = [1,2,3,4])
π_gpu = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)) |> gpu, actions = [1,2,3,4])

s = rand(2)
@test action(π_dqn, s) == argmax(π_dqn.Q(s))
@test action(π_gpu, s) == argmax(mdcall(π_gpu.Q,s, π_gpu.device))

@test π_dqn.Q(s) == π_dqn.Q⁻(s)

sb = rand(2,100)
@test size(value(π_dqn, sb)) == (4,100)
@test value(π_dqn, sb) == π_dqn.Q(sb)
@test size(value(π_gpu, sb)) == (4,100)
@test mean(abs.(value(π_gpu, sb) .- (value(π_gpu, sb |> gpu) |> cpu))) < 1e-7

## GaussianPolicy
p =  GaussianPolicy((s)-> zeros(Float32, 6), -0.5*ones(Float32, 6), cpu, Random.GLOBAL_RNG) 
a = [POMDPs.action(p, [1])[1] for i=1:100000]

@test abs(std(a) .- exp(-0.5)) < 1e-2
@test abs(mean(a)) < 1e-2


μ() = Chain(Dense(17, 64, tanh), Dense(64, 32, tanh), Dense(32, 1))
log_std() = -0.5*ones(Float32, 1)
p = GaussianPolicy(μ = μ(), logΣ = log_std())

d = MvNormal(μ()(rand(17)), exp.(log_std()))
std(rand(d, 10000))

s = rand(17)
meanval, logΣval = mdcall(p.μ, s, p.device), Crux.device(s)(p.logΣ)
d = MvNormal(meanval, exp.(logΣval))

a = [POMDPs.action(p, rand(Normal(0, .1), 17))[1] for i=1:100000]

@test abs(std(a) .- exp(-0.5)) < 1e-2
@test abs(mean(a)) < 1e-2



V() = Chain(Dense(17, 64, tanh), Dense(64, 32, tanh), Dense(32, 1))
p = ActorCritic(GaussianPolicy(μ = μ(), logΣ = log_std()), V())
a = [POMDPs.action(p, rand(Normal(0, 0.1), 17))[1] for i=1:100000]
@test abs(std(a) .- exp(-0.5)) < 1e-2
@test abs(mean(a)) < 1e-2

