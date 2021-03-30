using Crux
using Test
using POMDPs
import POMDPPolicies:FunctionPolicy
using Random
using Flux
using POMDPModels
using POMDPGym

mdp = GridWorldMDP()
pomdp = TigerPOMDP()

s = rand(initialstate(pomdp))
o = rand(initialobs(pomdp, s))

function POMDPs.gen(pomdp::TigerPOMDP, s, a)
    sp = rand(transition(pomdp, s, a ))
    o = rand(observation(pomdp, a, sp))
    r = reward(pomdp, s, a)
    (sp = sp, o=o, r=r)
end

π_explore = ϵGreedyPolicy(Crux.LinearDecaySchedule(1., 0.1, Int(100/2)), actions(mdp))

s1 = Sampler(mdp, FunctionPolicy((s) -> [:up]), A=DiscreteSpace(4, Symbol), max_steps=500)
s2 = Sampler(mdp, FunctionPolicy((s) -> [:up]), A=DiscreteSpace(4, Symbol), max_steps=50, π_explore = π_explore)
s3 = Sampler(pomdp, FunctionPolicy((s)-> 0), A=DiscreteSpace(length(actions(pomdp))))


## Steps! function
data = steps!(s1, Nsteps = 10)
@test size(data[:s], 2) == 10
@test all(data[:a] .== :up)
@test sum(data[:done]) > 0 || s1.episode_length == 10

data[:a]
data = steps!(s2, Nsteps = 10, explore = true)
@test size(data[:s], 2) == 10
@test !all(data[:a] .== :up)
@test sum(data[:done]) > 0 || s2.episode_length == 10

data = steps!(s2, Nsteps = 100)
@test s2.episode_length < 100

data = steps!(s3, Nsteps = 100)
@test size(data[:s], 2) == 100

## episodes! function
data, eps = episodes!(s2, Neps=10, return_episodes=true)

@test length(eps) == 10 

for e in eps
    @test e[1] < e[2]
    @test data[:done][e[2]] == 1f0 || e[2] - e[1] == s2.max_steps - 1
end

## positive returns only:
#TODO for GAIL

## Trajecory metrics
for e in eps
    ur =  undiscounted_return(data, e...)
    @test ur in [-10., 10., 0., 3., -5.]
    @test discounted_return(data, e..., discount(mdp)) ≈ ur*discount(mdp)^(e[2] - e[1])
    @test failure(data, e...) == (undiscounted_return(data, e...) < 0)
end

@test undiscounted_return(s1) < 3.
@test discounted_return(s1) < 3
@test failure(s1) < 1.

## fillto!
b = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4, Symbol), 100)
d = Dict(:s => 3*ones(2,3), :a => fill(:up, 1,3), :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3))
push!(b, d)
@test fillto!(b, s1, 3) == 0
@test fillto!(b, s1, 5) == 2
@test length(b) == 5

## Generalized Advantage Estimation
b = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4, Symbol), 100, [:advantage, :return])
d = Dict(:s => 3*ones(2,3), :a => fill(:up, 1,3), :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3), :return=>ones(1,3), :advantage=>ones(1,3))
push!(b, d)
push!(b, d)
#TODO: For policy gradient
fill_gae!(b, 1:5, ContinuousNetwork(Chain((s)->zeros(1, size(s, 2))), 1), 0.9f0, 0.7f0)
fill_returns!(b, 1:5, 0.7f0)

## Test sampling with a vector of mdps
mdps = [mdp, mdp, mdp]

samplers = Sampler(mdps, FunctionPolicy((s) -> [:up]), A=DiscreteSpace(4, Symbol))
@test length(samplers) == 3

data = steps!(samplers, Nsteps = 5)
data[:s]
@test size(data[:s],2) == 15

## Test episodes
mdp = GridWorldMDP()
s1 = Sampler(mdp, FunctionPolicy((s) -> [:up]), A=DiscreteSpace(4, Symbol), required_columns =[:episode_end], max_steps = 500,)
data, eps = episodes!(s1, Neps = 10, return_episodes = true,)
buffer = ExperienceBuffer(data)
eps2 = Crux.episodes(buffer)

@test all(eps .== eps2)

