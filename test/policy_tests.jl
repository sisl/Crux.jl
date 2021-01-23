using Crux
using Test
using Flux
using POMDPs
using POMDPGym
using POMDPPolicies
using Random
using Statistics
using Distributions

## Copyto and polyak average
c1 = Chain(Dense(5, 5, relu))
c2 = Chain(Dense(5, 5, relu))
c3 = Chain(Dense(5, 5, relu)) |> gpu

@test c1[1].W != c2[1].W
copyto!(c1, c2)

@test c1[1].W == c2[1].W

polyak_average!(c3, c2, 1f0)
@test c3[1].W isa CuArray
@test cpu(c3[1].W) == c2[1].W


## DiscreteNetwork 
π_dqn = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 4)), [1,2,3,4])
π_gpu = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 4)) |> gpu, [1,2,3,4])

s = rand(2)
@test action(π_dqn, s) == argmax(π_dqn.network(s))
@test action(π_gpu, s) == argmax(mdcall(π_gpu.network,s, π_gpu.device))

sb = rand(2,100)
@test size(value(π_dqn, sb)) == (4,100)
@test value(π_dqn, sb) == π_dqn.network(sb)
@test size(value(π_gpu, sb)) == (4,100)
@test mean(abs.(value(π_gpu, sb) .- (value(π_gpu, sb |> gpu) |> cpu))) < 1e-7

## GaussianPolicy
p =  GaussianPolicy(ContinuousNetwork((s)-> zeros(Float32, 6), 6), -0.5*ones(Float32, 6), cpu, Random.GLOBAL_RNG) 
a = [POMDPs.action(p, p, 0, [1])[1] for i=1:100000]

@test abs(std(a) .- exp(-0.5)) < 1e-2
@test abs(mean(a)) < 1e-2
