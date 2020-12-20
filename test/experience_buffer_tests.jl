using Crux
using POMDPModels
using Test
using DataStructures
using CUDA
using Flux
using Random

## data structures
@test MinHeap == MutableBinaryHeap{Float32, DataStructures.FasterForward}

## Fenwick Trees
t = FenwickTree(ones(Float32, 10))
@test t isa FenwickTree{Float32}
v = rand(16)
t2pow = FenwickTree(v)

# Inverse query on non power of 2 arrays
@test inverse_query(t, 0.5) == 1
@test inverse_query(t, 1.0) == 1
@test inverse_query(t, 0) == 1
@test inverse_query(t, -1) == 1
@test inverse_query(t, 3.0) == 3
@test inverse_query(t, 3.5) == 4
@test inverse_query(t, 4.0) == 4
@test inverse_query(t, 9.5) == 10
@test inverse_query(t, 10) == 10
@test inverse_query(t, 11) == 11

# power of 2 arrays
for i=1:1000
    r = 5*rand()
    ind = inverse_query(t2pow, r)

    @test prefixsum(t2pow, ind) ≥ r
    @test prefixsum(t2pow, ind-1) < r
end
@test inverse_query(t2pow, prefixsum(t2pow, 16)) == 16

# value test
@test all(t[i] == 1 for i=1:10)
@test all(t2pow[i] ≈ v[i] for i=1:16)

# update! test
update!(t, 4, 2.5)
@test t[4] ≈ 2.5
@test prefixsum(t, 4) ≈ 5.5


## Binary heap test
v = rand(19)
b = MutableBinaryHeap{Float64, DataStructures.FasterForward}(v)
@test first(b) == minimum(v)

push!(b, 1.0)
@test b[20] == 1.

update!(b, 1, 0.)
@test first(b) == 0.00

vlarge = rand(100000)
t = FenwickTree(vlarge)
h = MutableBinaryHeap{Float64, DataStructures.FasterForward}(vlarge)

# @time first(h)
# @time push!(h, 0.5)
# @time h[999]
# 
# @time t[999]
# @time inverse_query(t, 200)
# @time value(t, 999)


## mdp_data
d1 = mdp_data(3, 4, 100)
d2 = mdp_data(3, 4, 100, gae = true)
d3 = mdp_data(3, 4, 100, ArrayType = CuArray)

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
b_gpu = b |> gpu

@test b isa ExperienceBuffer{Array}
@test b_gpu isa ExperienceBuffer{CuArray}

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
@test Crux.device(b) == cpu
@test Crux.device(bpriority) == cpu
@test Crux.device(b_gpu) == gpu

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
d = Dict(:s => 3*ones(2,3), :a => ones(4,3), :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3))
push!(b, d)
@test length(b) == 4
@test b[:s][:,2:end] ==  3*ones(2,3)
@test b[:a][:,2:end] == ones(4,3)
@test b[:sp][:,2:end] == 5*ones(2,3)
@test b[:r][:,2:end] == 6*ones(1,3)
@test b[:done][:,2:end] == ones(1,3)

# push a buffer
push!(b, b)
@test length(b) == 8
for k in keys(b)
    @test b[k][:, 1:4] == b[k][:, 5:8]
end

## minibatch
I = [1,2,4]
d = minibatch(b, I)
for k in keys(d)
    @test all(d[k] .== b[k][:, I])
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

## Multi-D states
b2d = ExperienceBuffer((2,2), 4, 100; S = UInt8)
@test ndims(b2d[:s]) == 3

d = Dict(:s => 3*ones(2,2,1), :a => ones(4,1), :sp => ones(2,2,1), :r => ones(1,1), :done => zeros(1,1))
push!(b2d, d)

@test all(b2d[:s] .== 3)

