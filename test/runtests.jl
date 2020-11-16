include("../src/includes.jl")
using Test
using POMDPModels

## Test Utils
vgpu = CuArray{Float32, 2}(undef, 10, 10)
@test device(vgpu) == gpu

vcpu = Array{Float32, 2}(undef, 10, 10)
@test device(vcpu) == cpu

## Test Experience Buffers
b = ExperienceBuffer(2, 4, 100, gae = true)
b_gpu = empty_like(b, device = gpu)

@test b isa ExperienceBuffer{Array{Float32, 2}}
@test b_gpu isa ExperienceBuffer{CuArray{Float32, 2}}

@test length(keys(b.data)) == length(keys(b_gpu.data))
@test :s in keys(b_gpu.data)
@test :a in keys(b_gpu.data)
@test :sp in keys(b_gpu.data)
@test :r in keys(b_gpu.data)
@test :done in keys(b_gpu.data)

@test size(b[:s]) == (2,0)
@test length(b) == 0
@test capacity(b) == 100

@test size(b_gpu[:s]) == (2,0)
@test length(b_gpu) == 0
@test capacity(b_gpu) == 100

push!(b, Dict(:s =>rand(2), :a => rand(4), :s => rand(2), :r => 1., :done => 1., :advantage => 1., :return => :1))

mdp = SimpleGridWorld(size = (10,10), tprob = .7)
push!(b, rand(2), :up,1., rand(2), 1., mdp)
@test length(b) == 2
@test b.next_ind == 3

clear!(b)
@test length(b) == 020
@test b.next_ind == 1

@test isgpu(b_gpu)
@test !isgpu(b)

function get_indices(b, start, Nsteps)
    stop = mod1(start+Nsteps-1, length(b))
    Nsteps > length(b) && (start = stop+1) # Handle overlap
    (stop > start) ? collect(start:stop) : [start:length(b)..., 1:stop...]
end

fill!(b, mdp, RandomPolicy(mdp))
@test length(b) == 100
@test get_indices(b, 4, 60) == [4:63 ...]
@test get_indices(b, 1, 100) == [1:100 ...]
@test get_indices(b, 1, 101) == [2:100 ..., 1]
@test get_indices(b, 1, 120) == [21:100 ..., 1:20 ...]
@test get_indices(b, 90, 20) == [90:100 ..., 1:9 ...]

baseline = Baseline(V = Chain(Dense(2,32, relu), Dense(32, 1)))
fill_gae!(b, 1, 100, baseline.V, 0.9, 0.7)
fill_returns!(b, 1, 100, 0.7)

## Policy tests
Qnet = Chain(Dense(2,32, relu), Dense(32, 4))
π = CategoricalPolicy(Qnet, mdp; device = cpu)

@test size(value(π, b[:s])) == (4,100)

y = target(Qnet, b, 0.9)
sum(value(π, b[:s]) .* b[:a], dims = 1)

## Logging
@test elapsed(1000, 100) 
@test elapsed(47501, 500, last_i = 47001)


