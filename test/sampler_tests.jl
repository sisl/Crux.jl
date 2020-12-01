using Shard
using Test
using POMDPs
using POMDPPolicies
using Random
using Flux
include("../examples/mdps/gridworld.jl")

mdp = SimpleGridWorld()
pomdp = TigerPOMDP()

function POMDPs.gen(pomdp::TigerPOMDP, s, a, rng = Random.GLOBAL_RNG)
    sp = rand(rng, transition(pomdp, s, a ))
    o = rand(rng, observation(pomdp, a, sp))
    r = reward(pomdp, s, a)
    (sp = sp, o=o, r=r)
end

exploration_policy = EpsGreedyPolicy(LinearDecaySchedule(start=1., stop=0.1, steps=100/2), Random.GLOBAL_RNG, actions(mdp))

s1 = Sampler(mdp, FunctionPolicy((s) -> :up), 2, 4, max_steps = 500)
s2 = Sampler(mdp, FunctionPolicy((s) -> :up), 2, 4, max_steps = 50, exploration_policy = exploration_policy)
s3 = Sampler(pomdp, FunctionPolicy((s)-> 0), sdim(pomdp), adim(pomdp))

## Steps! function
data = steps!(s1, Nsteps = 10)
@test size(data[:s], 2) == 10
@test sum(data[:a][1, :]) == 10
@test sum(data[:done]) > 0 || s1.episode_length == 10


data = steps!(s2, Nsteps = 10, explore = true)
@test size(data[:s], 2) == 10
@test sum(data[:a][1, :]) < 10
@test sum(data[:done]) > 0 || s2.episode_length == 10

data = steps!(s2, Nsteps = 100)
@test s2.episode_length < 100

data = steps!(s3, Nsteps = 100)
@test size(data[:s], 2) == 100

## episodes! function
data, episodes = episodes!(s2, Neps = 10, return_episodes = true)

@test length(episodes) == 10 

for e in episodes
    @test e[1] < e[2]
    @test data[:done][e[2]] == 1f0 || e[2] - e[1] == s2.max_steps - 1
end

## positive returns only:
#TODO for GAIL

## Trajecory metrics
for e in episodes
    ur =  undiscounted_return(data, e...)
    @test ur in [-10., 10., 0., 3., -5.]
    @test discounted_return(data, e..., discount(mdp)) ≈ ur*discount(mdp)^(e[2] - e[1])
    @test failure(data, e...) == (undiscounted_return(data, e...) < 0)
end

@test undiscounted_return(s1) < 3.
@test discounted_return(s1) < 3
@test failure(s1) < 1.

## fillto!
b = ExperienceBuffer(2, 4, 100, gae = true)
d = Dict(:s => 3*ones(2,3), :a => 4*ones(4,3), :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3))
push!(b, d)

@test fillto!(b, s1, 3) == 0
@test fillto!(b, s1, 5) == 2
@test length(b) == 5

## Generalized Advantage Estimation
#TODO: For policy gradient
baseline = Baseline(Chain(Dense(2,32, relu), Dense(32, 1)))
fill_gae!(b, 1:5, baseline.V, 0.9f0, 0.7f0)
fill_returns!(b, 1:5, 0.7f0)

## Test sampling with a vector of mdps
mdps = [mdp, mdp, mdp]

samplers = Sampler(mdps, FunctionPolicy((s) -> :up), 2, 4)
@test length(samplers) == 3

data = steps!(samplers, Nsteps = 5)
data[:s]
@test size(data[:s],2) == 15

