include("../src/includes.jl")
using POMDPModels
using Test

## data structures
@test MinHeap == MutableBinaryHeap{Float32, DataStructures.FasterForward}

## mdp_data
d1 = mdp_data(3, 4, 100)
d2 = mdp_data(3, 4, 100, gae = true)
d3 = mdp_data(3, 4, 100, Atype = CuArray{Float32, 2})

@test size(d1[:s]) == (3, 100) && size(d2[:s]) == (3, 100) && size(d3[:s]) == (3, 100)
@test size(d1[:sp]) == (3, 100) && size(d2[:sp]) == (3, 100) && size(d3[:sp]) == (3, 100)
@test size(d1[:a]) == (4, 100) && size(d2[:a]) == (4, 100) && size(d3[:a]) == (4, 100)
@test size(d1[:r]) == (1, 100) && size(d2[:r]) == (1, 100) && size(d3[:r]) == (1, 100)
@test size(d1[:done]) == (1, 100) && size(d2[:done]) == (1, 100) && size(d3[:done]) == (1, 100)
@test size(d2[:return]) == (1, 100) && size(d2[:advantage]) == (1, 100)
@test !haskey(d1, :return) && !haskey(d1, :advantage)
@test d3[:s] isa CuArray

## circular_indices
@test circular_indices(4, 60, 100) == [4:63 ...]
@test circular_indices(1, 100, 100) == [1:100 ...]
@test circular_indices(1, 101, 100) == [2:100 ..., 1]
@test circular_indices(1, 120, 100) == [21:100 ..., 1:20 ...]
@test circular_indices(90, 20, 100) == [90:100 ..., 1:9 ...]

## Construction
b = ExperienceBuffer(2, 4, 100, gae = true)
bpriority = ExperienceBuffer(2, 4, 50, prioritized = true, gae = true)
b_gpu = ExperienceBuffer(b, device = gpu)

@test b isa ExperienceBuffer{Array{Float32, 2}}
@test b_gpu isa ExperienceBuffer{CuArray{Float32, 2}}

@test length(keys(b.data)) == length(keys(b_gpu.data))
@test :s in keys(b_gpu.data)
@test :a in keys(b_gpu.data)
@test :sp in keys(b_gpu.data)
@test :r in keys(b_gpu.data)
@test :done in keys(b_gpu.data)
@test :weight in keys(b_gpu.data)

## Base functions 
@test keys(b) == keys(b.data)

@test size(b[:s]) == (2,0)
@test length(b) == 0
@test capacity(b) == 100

@test size(b_gpu[:s]) == (2,0)
@test length(b_gpu) == 0
@test capacity(b_gpu) == 100

@test prioritized(bpriority)
@test device(b) == cpu
@test device(bpriority) == cpu
@test device(b_gpu) == gpu

## push!
#push dictionary with one element
d = Dict(:s => 2*ones(2,1), :a => ones(4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(b, d)
@test length(b) == 1
@test b[:s] == 2*ones(2,1)
@test b[:a] == ones(4,1)
@test b[:sp] == ones(2,1)
@test b[:r] == ones(1,1)
@test b[:done] == zeros(1,1)

# push dictionary with more than one element
d = Dict(:s => 3*ones(2,3), :a => 4*ones(4,3), :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3))
push!(b, d)
@test length(b) == 4
@test b[:s][:,2:end] ==  3*ones(2,3)
@test b[:a][:,2:end] == 4*ones(4,3)
@test b[:sp][:,2:end] == 5*ones(2,3)
@test b[:r][:,2:end] == 6*ones(1,3)
@test b[:done][:,2:end] == ones(1,3)

# push a buffer
push!(b, b)
@test length(b) == 8
for k in keys(b)
    @test b[k][:, 1:4] == b[k][:, 5:8]
end

## update_priorities!
update_priorities!(bpriority, [1,2,3], [1., 2., 3.])
@test bpriority.max_priority == 3.0
@test bpriority.priorities[1] ≈ 1f0^bpriority.α
@test bpriority.priorities[2] ≈ 2f0^bpriority.α
@test bpriority.priorities[3] ≈ 3f0^bpriority.α
@test bpriority.minsort_priorities[1] ≈ 1f0^bpriority.α
@test bpriority.minsort_priorities[2] ≈ 2f0^bpriority.α
@test bpriority.minsort_priorities[3] ≈ 3f0^bpriority.α

push!(bpriority, d)
push!(bpriority, d)
@test bpriority.max_priority == 3.0
for i=1:6
    @test bpriority.priorities[i] ≈ 3f0^bpriority.α
    @test bpriority.minsort_priorities[i] ≈ 3f0^bpriority.α
end
    

## sampling

# uniform sample
t = ExperienceBuffer(2, 4, 10, gae = true)
rand!(Random.GLOBAL_RNG, t, b)

t = ExperienceBuffer(2, 4, 3, gae = true)
rng = MersenneTwister(0)
ids = rand(rng, 1:length(b), 3)
rand!(MersenneTwister(0), t, b)

for k in keys(t)
    @test t[k] == b[k][:,ids]
end

# Test the multi-buffer sampling
t1 = ExperienceBuffer(2, 4, 10)
d = Dict(:s => ones(2,1), :a => ones(4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t1, d)

t2 = ExperienceBuffer(2, 4, 10)
d = Dict(:s => 2*ones(2,1), :a => ones(4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t2, d)

t3 = ExperienceBuffer(2, 4, 10)
d = Dict(:s => 3*ones(2,1), :a => ones(4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t3, d)

t = ExperienceBuffer(2, 4, 10, gae = true)
rand!(Random.GLOBAL_RNG, t, t1, t2, t3)

@test all(t[:s][:, 1:4] .== 1.0)
@test all(t[:s][:, 5:7] .== 2.0)
@test all(t[:s][:, 8:10] .== 3.0)



# Priority samples
bpriority[:s] .= rand(Float32, 2, 6)
update_priorities!(bpriority, [1:6...], [1.:6. ...])
t = ExperienceBuffer(2, 4, 1000, gae=true)
priorities = [bpriority.priorities[i] for i=1:length(bpriority)]

rand!(rng, t, bpriority)

# Test the the frequency of samples is proportional to their priority
length(t)
freqs = [sum(t.indices .== i) for i=1:6] ./ length(t)
probs = priorities ./ sum(priorities)
relerr = abs.(freqs .- probs) ./ probs
@test all(relerr .< 0.01)

@test all(t[:s] .== bpriority[:s][:,t.indices])
@test all(t[:weight] .<= 1.)

## merge 
# bmerge = merge(b, b, capacity = 300)
# @test length(bmerge) == 200
# @test capacity(bmerge) == 300
# @test bmerge[:s][:, 1:100] == bmerge[:s][:, 101:200] 

