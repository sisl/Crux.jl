include("../src/includes.jl")
using POMDPModels
using Test

## Experience Buffer
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

@test keys(b) == keys(b.data)

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
@test length(b) == 0
@test b.next_ind == 1

@test isgpu(b_gpu)
@test !isgpu(b)

fill!(b, mdp, policy = RandomPolicy(mdp))
@test length(b) == 100
@test circular_indices(4, 60, length(b)) == [4:63 ...]
@test circular_indices(1, 100, length(b)) == [1:100 ...]
@test circular_indices(1, 101, length(b)) == [2:100 ..., 1]
@test circular_indices(1, 120, length(b)) == [21:100 ..., 1:20 ...]
@test circular_indices(90, 20, length(b)) == [90:100 ..., 1:9 ...]

baseline = Baseline(Chain(Dense(2,32, relu), Dense(32, 1)))
fill_gae!(b, 1, 100, baseline.V, 0.9f0, 0.7f0)
fill_returns!(b, 1, 100, 0.7f0)

bmerge = merge(b, b, capacity = 300)
@test length(bmerge) == 200
@test capacity(bmerge) == 300
@test bmerge[:s][:, 1:100] == bmerge[:s][:, 101:200] 

