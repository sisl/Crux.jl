using Crux
using Test
using Flux
using POMDPs
using POMDPGym
using Random
using Statistics
using Distributions
using CUDA

## Deep copy of gpu policies
π1 = ContinuousNetwork(Chain(Dense(5, 5, relu))) |> gpu
π1⁻ = deepcopy(π1)

polyak_average!(π1, π1⁻, 0.005f0)

weightbefore = cpu(π1⁻.network[1].weight)

π1.network[1].weight .= 1

@test cpu(π1⁻.network[1].weight)  == weightbefore




## Copyto and polyak average
c1 = Chain(Dense(5, 5, relu))
c2 = Chain(Dense(5, 5, relu))
c3 = Chain(Dense(5, 5, relu)) |> gpu

@test c1[1].weight != c2[1].weight
copyto!(c1, c2)

@test c1[1].weight == c2[1].weight

polyak_average!(c3, c2, 1f0)
USE_CUDA && @test c3[1].weight isa CuArray
@test cpu(c3[1].weight) == c2[1].weight


## Continuous Network
p = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)))

@test Crux.device(p) == cpu
@test p.output_dim == 4

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
@test p_gpu.output_dim == 4

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

p2 = ContinuousNetwork(ConstantLayer(rand(10)), 10)
@test size(p2(rand(1))) == (10,)

## Discrete Network
p = DiscreteNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)), [1,2, 3, 4])

@test Crux.device(p) == cpu
@test p.outputs == [1,2,3,4]

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
@test p_gpu.outputs == [1,2,3,4]

@test length(Flux.params(p)) == 4
@test length(layers(p)) == 2

s0 = rand(2)
s = rand(2,100)
a0 = rand([true, false], 4)
a = rand([true, false], 4, 100)

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
@test all(logpdf(p, s, Flux.onehotbatch(a[:], [1,2,3,4])) .≈ logprob)

@test size(entropy(p, s0)) == (1,)
@test size(entropy(p, s)) == (1, 100)

@test action_space(p).vals == DiscreteSpace(4).vals

# Onhotbatch - This used to error
Flux.onehotbatch(p_gpu, ones(Int, 1, 10) |> gpu)

# Testing the logit conversion function
p = DiscreteNetwork(Chain(Dense(2, 32, relu), Dense(32, 4, sigmoid)), [1,2, 3, 4], logit_conversion=(π, s) -> value(π, s) ./ sum(value(π, s),dims=1))

vals = value(p, s)
lgts = Crux.logits(p, s)
@test sum(lgts) ≈ 100



## Mixture Network
Q1 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 3)), [1,2,3])
Q2 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 3)), [1,2,3])
p = MixtureNetwork([Q1, Q2], [0.1, 0.9])
s = rand(2,100)
a = rand([true, false], 3, 100)


@test length(Flux.trainable(p)) == 3
@test length(layers(p)) == 5
@test Crux.device(p) == cpu

@test action_space(p) == action_space(p.networks[1])

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
USE_CUDA && @test all([Crux.device(n)== gpu for n in p_gpu.networks])

@test length(Flux.params(p)) == 9
@test length(layers(p)) == 5

# @test all(Crux.value(p, s) .≈ Crux.value(p_gpu, s))
# @test Crux.valueall(p, s) == [Crux.value(p.networks[1], s), Crux.value(p.networks[2], s)]
# @test Crux.valueall(p, s, a) == [Crux.value(p.networks[1], s, a), Crux.value(p.networks[2], s, a)]
@test_broken try # I think this is because the mixture network isn't designed for batch actions?
	action(p, s)
	true
catch
	false
end


## Double Network
Q1 = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 1)))
Q2 = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 2)))
p = DoubleNetwork(Q1, Q2)

@test action_space(p) == action_space(p.N1)

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
USE_CUDA && @test Crux.device(p_gpu.N1) == gpu
USE_CUDA && @test Crux.device(p_gpu.N2) == gpu

@test length(Flux.params(p)) == 8
@test length(layers(p)) == 4

@test all(value(p, s) .≈ value(p_gpu, s))
@test value(p, s) == (value(p.N1, s), value(p.N2, s))
@test value(p, s[1:1,:], s[2:2,:]) == (value(p.N1, s[1:1,:], s[2:2,:]), value(p.N2, s[1:1,:], s[2:2,:]))
@test action(p, s) == (action(p.N1, s), action(p.N2, s))

Q1 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 1)), [1,2])
Q2 = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 2)), [1,2])
p = DoubleNetwork(Q1, Q2)

Random.seed!(1)
e1 = exploration(p, s)
Random.seed!(1)
e2 = (exploration(p.N1, s), exploration(p.N2, s))
@test e1 == e2

a = rand([1,], 100)
@test logpdf(p, s, Flux.onehotbatch(Q1, a)) == (logpdf(p.N1, s, Flux.onehotbatch(Q1, a)), logpdf(p.N2, s, Flux.onehotbatch(Q2, a)))
@test entropy(p, s) == (entropy(p.N1, s), entropy(p.N2, s))

@test action_space(p) == action_space(p.N1)

## Actor Critic
Anet = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 4)))
C = ContinuousNetwork(Chain(Dense(6,32, relu), Dense(32, 1)))
p = ActorCritic(Anet, C)

@test action_space(p) == action_space(p.A)

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
USE_CUDA && @test Crux.device(p_gpu.A) == gpu
USE_CUDA && @test Crux.device(p_gpu.C) == gpu

@test length(Flux.params(p)) == 8
@test length(layers(p)) == 4

sfull = rand(6, 100)
a = rand(4, 100)
@test all(value(p, sfull) .≈ value(p_gpu, sfull))
@test value(p, sfull) == value(p.C, sfull)
@test value(p, s, a) == value(p.C, s, a)
@test action(p, s) == action(p.A, s)
@test all(action(p, s) .≈ action(p_gpu, s))

Anet = DiscreteNetwork(Chain(Dense(2,32, relu), Dense(32, 1)), [1,])
p = ActorCritic(Anet, C)

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
μnet = ContinuousNetwork(Chain(Dense(2,32, relu), Dense(32, 4)))
logΣnet = ones(Float32, 4)
p = GaussianPolicy(μnet, logΣnet)
@test Crux.device(p) == cpu
@test all(value(p.logΣ, rand(2)) .== 1)

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
USE_CUDA && @test Crux.device(p_gpu.μ) == gpu
USE_CUDA && @test Crux.device(p_gpu.logΣ) == gpu
@test all(value(p.logΣ, rand(2)) .== 1)

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
μnet = ContinuousNetwork(Chain(trunk..., Dense(32, 4)))
logΣnet = ContinuousNetwork(Chain(trunk..., Dense(32, 4)))
p = SquashedGaussianPolicy(μnet, logΣnet)
@test Crux.device(p) == cpu

p_gpu = p |> gpu
USE_CUDA && @test Crux.device(p_gpu) == gpu
USE_CUDA && @test Crux.device(p_gpu.μ) == gpu
USE_CUDA && @test Crux.device(p_gpu.logΣ) == gpu

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


## Distribution Network
cmv = DistributionPolicy(Distributions.product_distribution([Normal(-1,1) for i=1:6]))
cuv = DistributionPolicy(Normal(-1,1))
duv = DistributionPolicy(Categorical([0.1, 0.2, 0.3, 0.4]))
doc = DistributionPolicy(ObjectCategorical([:up, :down]))

@test action_space(cmv) == ContinuousSpace(6)
@test action_space(cuv) == ContinuousSpace(1)
@test action_space(duv) == DiscreteSpace(4, Base.OneTo(4))
@test action_space(doc).N ==2
@test action_space(doc).vals == [:up, :down]
ps = [cmv, cuv, duv, doc]

@test all(Crux.device.(ps) .== cpu)

@test all(length.(Flux.params.(ps)) .== 0)
@test all(length.(layers.(ps)) .== 0)

s0 = rand(2)
s = rand(2,100)

# Check exploration and logprob
for p in ps
	@test size(action(p, s0)) == (length(p.distribution),)
	@test size(action(p, s)) == (length(p.distribution),100)

	global a, logprob = exploration(p, s0)
	@test size(a) == (length(p.distribution),)
	@test size(logprob) == (1,)
	@test all(logpdf(p, s0, a) .≈ logprob)

	global a, logprob = exploration(p, s)
	@test size(a) == (length(p.distribution),100)
	@test size(logprob) == (1,100)
	@test all(logpdf(p, s, a) .≈ logprob)

	@test size(entropy(p, s0)) == (1,)
	@test size(entropy(p, s)) == (1, 100)
end

# Check that passing by one hot encoding works
p = duv
a0 = action(p, s0)
a = action(p, s)
ls0 = logpdf(p, s0, a0)
ls = logpdf(p, s0, a)

a0 = Flux.onehotbatch(a0, [1,2,3,4])
a = Flux.onehotbatch(a[:], [1,2,3,4])
ls0_oh = logpdf(p, s0, a0)
ls_oh = logpdf(p, s0, a)

@test ls0 == ls0_oh
@test ls == ls_oh


## ϵGreedyPolicy
p_on = DiscreteNetwork(Dense(2,2), [2,3])
p = ϵGreedyPolicy(0f0, [2,3])
a, logprob = exploration(p, s0; π_on=p_on, i=1)

@test a[1] in [2,3]
@test logprob[1] <= 0


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
