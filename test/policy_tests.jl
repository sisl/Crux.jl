using Crux
using Test
using Flux
using POMDPs
using POMDPGym
using Random
using Statistics
using Distributions
using CUDA

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

## Continuous Network
p = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)))

@test Crux.device(p) == cpu
@test p.output_dim == (4,)

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test p_gpu.output_dim == (4,)

@test length(Flux.params(p)) == 4
@test length(layers(p)) == 2

s0 = rand(2)
s = rand(2,100)

@test size(value(p, s0)) == (4,)
@test size(value(p, s)) == (4,100)
@test size(value(p, s[1:1,:], s[2:2,:])) == (4, 100)
@test size(value(p, s0[1], s[2])) == (4,)

@test size(action(p, s0)) == (4,)
@test size(action(p, s)) == (4,100)
@test all(action(p, s) .≈ action(p_gpu, s))

@test value(p, s0) == action(p, s0)
@test value(p, s) == action(p, s)

@test action_space(p) == ContinuousSpace(4)

## Discrete Network
p = DiscreteNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)), [1,2, 3, 4])

@test Crux.device(p) == cpu
@test p.outputs == [1,2,3,4]

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test p_gpu.outputs == [1,2,3,4]

@test length(Flux.params(p)) == 4
@test length(layers(p)) == 2

s0 = rand(2)
s = rand(2,100)
a0 = rand([1,2,3,4],1)
a = rand([1,2,3,4], 100)

@test size(value(p, s0)) == (4,)
@test size(value(p, s)) == (4,100)

@test size(value(p, s0, a0)) == (1,)
@test size(value(p, s, a)) == (1, 100)

@test size(action(p, s0)) == (1,)
@test size(action(p, s)) == (1,100)
@test all(action(p, s) .≈ action(p_gpu, s))

a, logprob = exploration(p, s0)
@test size(a) == (1,)
@test size(logprob) == (1,)
@test all(logpdf(p, s0, a) .≈ logprob)

a, logprob = exploration(p, s)
@test size(a) == (1,100)
@test size(logprob) == (1,100)
@test all(logpdf(p, s, a) .≈ logprob)

@test size(entropy(p, s0)) == (1,)
@test size(entropy(p, s)) == (1, 100)

@test action_space(p) == DiscreteSpace(4, Int)

## Double Network
Q1 = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 1)))
Q2 = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 2)))
p = DoubleNetwork(Q1, Q2)

@test action_space(p) == action_space(p.N1)

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test Crux.device(p_gpu.N1) == gpu
@test Crux.device(p_gpu.N2) == gpu

@test length(Flux.params(p)) == 8
@test length(layers(p)) == 4

@test all(value(p, s) .≈ value(p_gpu, s))
@test value(p, s) == (value(p.N1, s), value(p.N2, s))
@test value(p, s[1:1,:], s[2:2,:]) == (value(p.N1, s[1:1,:], s[2:2,:]), value(p.N2, s[1:1,:], s[2:2,:]))
@test action(p, s) == (action(p.N1, s), action(p.N2, s))

Q1 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 1)), [1,])
Q2 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 2)), [1,2])
p = DoubleNetwork(Q1, Q2)

Random.seed!(1)
e1 = exploration(p, s)
Random.seed!(1)
e2 = (exploration(p.N1, s), exploration(p.N2, s))
@test e1 == e2 

a = rand([1,], 100)
@test logpdf(p, s, a) == (logpdf(p.N1, s, a), logpdf(p.N2, s, a))
@test entropy(p, s) == (entropy(p.N1, s), entropy(p.N2, s))

@test action_space(p) == action_space(p.N1)

## Actor Critic
A = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 4)))
C = ContinuousNetwork(Chain(Dense(6,32, relu), Dense(32, 1)))
p = ActorCritic(A, C)

@test action_space(p) == action_space(p.A)

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test Crux.device(p_gpu.A) == gpu
@test Crux.device(p_gpu.C) == gpu

@test length(Flux.params(p)) == 8
@test length(layers(p)) == 4

sfull = rand(6, 100)
a = rand(4, 100)
@test all(value(p, sfull) .≈ value(p_gpu, sfull))
@test value(p, sfull) == value(p.C, sfull)
@test value(p, s, a) == value(p.C, s, a)
@test action(p, s) == action(p.A, s)
@test all(action(p, s) .≈ action(p_gpu, s))

A = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 1)), [1,])
p = ActorCritic(A, C)

Random.seed!(1)
e1 = exploration(p, s)
Random.seed!(1)
e2 = exploration(p.A, s)
@test e1 == e2 

a = rand([1,], 100)
@test logpdf(p, s, a) == logpdf(p.A, s, a)
@test entropy(p, s) == entropy(p.A, s)

@test action_space(p) == action_space(p.A)


## Gaussian Policy
μ = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 4)))
logΣ = ones(Float32, 4)
p = GaussianPolicy(μ, logΣ)
@test Crux.device(p) == cpu
@test all(p.logΣ .== 1)

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test Crux.device(p_gpu.μ) == gpu
@test Crux.device(p_gpu.logΣ) == gpu
@test all(p.logΣ .== 1)

@test length(Flux.params(p)) == 5
@test length(layers(p)) == 3

s0 = rand(2)
s = rand(2,100)
a0 = rand([1,2,3,4],1)
a = rand([1,2,3,4], 100)

@test size(action(p, s0)) == (4,)
@test size(action(p, s)) == (4,100)
@test all(action(p, s) .≈ action(p_gpu, s))

a, logprob = exploration(p, s0)
@test size(a) == (4,)
@test size(logprob) == (1,)
@test all(logpdf(p, s0, a) .≈ logprob)

a, logprob = exploration(p, s)
@test size(a) == (4,100)
@test size(logprob) == (1,100)
@test all(logpdf(p, s, a) .≈ logprob)

@test size(entropy(p, s0)) == ()
@test size(entropy(p, s)) == ()

@test action_space(p) == ContinuousSpace(4)

# Check empirically if the mean and std are correct
p =  GaussianPolicy(ContinuousNetwork((s)-> zeros(Float32, 6), 6), -0.5*ones(Float32, 6)) 
a = [exploration(p, [1])[1][1] for i=1:100000]

@test abs(std(a) .- exp(-0.5)) < 1e-2
@test abs(mean(a)) < 1e-2

## Squashed Gaussian Policy
trunk = Chain(Dense(2,32, relu))
μ = ContinuousNetwork(Chain(trunk..., Dense(32, 4)))
logΣ = ContinuousNetwork(Chain(trunk..., Dense(32, 4)))
p = SquashedGaussianPolicy(μ, logΣ)
@test Crux.device(p) == cpu

p_gpu = p |> gpu
@test Crux.device(p_gpu) == gpu
@test Crux.device(p_gpu.μ) == gpu
@test Crux.device(p_gpu.logΣ) == gpu

@test length(Flux.params(p)) == 6
@test length(layers(p))  == 3


s0 = rand(2)
s = rand(2,100)

@test size(action(p, s0)) == (4,)
@test size(action(p, s)) == (4,100)
@test all(action(p, s) .≈ action(p_gpu, s))

a, logprob = exploration(p, s0)
@test size(a) == (4,)
@test size(logprob) == (1,)
@test all(logpdf(p, s0, a) .≈ logprob)

a, logprob = exploration(p, s)
@test all(a .>= -1)
@test all(a .<= 1)
@test size(a) == (4,100)
@test size(logprob) == (1,100)
@test all(logpdf(p, s, a) .≈ logprob)

@test size(entropy(p, s0)) == (1,)
@test size(entropy(p, s)) == (1, 100)

@test action_space(p) == ContinuousSpace(4)

## ϵGreedyPolicy
p_on = DiscreteNetwork((s)->[0.5, 1.0], [2,3])
p = ϵGreedyPolicy(0f0, [1,])
a, logprob = exploration(p, s0; π_on=p_on, i=1)
@test a==[3] && isnan(logprob)

p = ϵGreedyPolicy(1f0, [1,])
a, logprob = exploration(p, s0; π_on=p_on, i=1)
@test a==[1] && isnan(logprob)

p = ϵGreedyPolicy(LinearDecaySchedule(1f0, 0f0, 10), [1,])
@test exploration(p, s0; π_on=p_on, i=0)[1] == [1]
@test exploration(p, s0; π_on=p_on, i=10)[1] == [3]
@test p.ϵ(5) == 0.5


## GaussianNoiseExplorationPolicy
p_on = ContinuousNetwork((s)->[1.0], 1)
p = GaussianNoiseExplorationPolicy(10f0, ϵ_min=-0.1, ϵ_max=0.1)
a, logprob = exploration(p, s0; π_on=p_on, i=1)
@test any(a[1] .≈ [0.9, 1.1]) && isnan(logprob)
@test p.σ(1) == 10f0
@test p.a_min == [-Inf32]
@test p.a_max == [Inf32]
@test p.ϵ_min == -0.1f0
@test p.ϵ_max == 0.1f0

p = GaussianNoiseExplorationPolicy((i)->10f0, a_min=[-0.1], a_max=[0.1])
a, logprob = exploration(p, s0; π_on=p_on, i=1)
@test any(a[1] .≈ [-0.1, .1]) && isnan(logprob)
@test p.σ(1) == 10f0
@test p.ϵ_min == -Inf32
@test p.ϵ_max == Inf32
@test p.a_min == [-0.1f0]
@test p.a_max == [0.1f0]

## First exlore policy
p = FirstExplorePolicy(10, ContinuousNetwork((x)->[0,], 1))
a, logprob = exploration(p, s; π_on=ContinuousNetwork((x)->[1,], 1), i=2)
@test a==[0]

a, logprob = exploration(p, s; π_on=ContinuousNetwork((x)->[1,], 1), i=20)
@test a==[1]

