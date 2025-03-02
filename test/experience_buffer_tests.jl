using Crux
using Test
using CUDA
using Flux
using Random

## mdp_data
d1 = mdp_data(ContinuousSpace(3), ContinuousSpace(4), 100)
d2 = mdp_data(ContinuousSpace(3), ContinuousSpace(4), 100, [:weight, :t, :advantage, :return, :logprob])
# @test_throws ErrorException  mdp_data(ContinuousSpace(3), ContinuousSpace(4), 100, [:bad_key]) # Now this just throws a warning
d3 = mdp_data(ContinuousSpace(3), ContinuousSpace(4), 100, ArrayType = USE_CUDA ? CuArray : Array)

@test d1[:s] == zeros(Float32, 3, 100) && d2[:s] == zeros(Float32, 3, 100) && size(d3[:s]) == (3, 100)
@test d1[:sp] == zeros(Float32, 3, 100) && d2[:sp] == zeros(Float32, 3, 100) && size(d3[:sp]) == (3, 100)
@test d1[:a] == zeros(Bool, 4, 100) && size(d2[:a]) == (4, 100) && size(d3[:a]) == (4, 100)
@test d1[:r] == zeros(Float32, 1, 100) && size(d2[:r]) == (1, 100) && size(d3[:r]) == (1, 100)
@test d1[:done] == zeros(Bool, 1, 100) && size(d2[:done]) == (1, 100) && size(d3[:done]) == (1, 100)
@test d2[:return] == zeros(1, 100) && d2[:advantage] == zeros(1, 100) && d2[:logprob] == zeros(1, 100) && d2[:t] == zeros(1, 100) && d2[:weight] == ones(1, 100)
@test !haskey(d1, :return) && !haskey(d1, :advantage) && !haskey(d1, :logprob) && !haskey(d1, :t) && !haskey(d1, :weight)
USE_CUDA && @test d3[:s] isa CuArray

## circular_indices
circ_inds(start, Nsteps, l) = mod1.(start:start + Nsteps - 1, l)
@test circ_inds(4, 60, 100) == [4:63 ...]
@test circ_inds(1, 100, 100) == [1:100 ...]
@test circ_inds(1, 101, 100) == [1:100 ..., 1]
@test circ_inds(1, 120, 100) == [1:100 ..., 1:20 ...]
@test circ_inds(90, 20, 100) == [90:100 ..., 1:9 ...]

## Get last N indices test
# A not full buffer
b = ExperienceBuffer(ContinuousSpace(2), ContinuousSpace(1), 100)
d = Dict(:s => 2*ones(2,50), :a => ones(Bool, 1,50), :sp => ones(2,50), :r => ones(1,50), :done => zeros(1,50), :weight=>zeros(1,50))
push!(b, d)


@test get_last_N_indices(b, 10) == collect(41:50)
@test get_last_N_indices(b, 1) == [50]
@test get_last_N_indices(b, 50) == collect(1:50)
@test get_last_N_indices(b, 51) == collect(1:50)
@test get_last_N_indices(b, 1000) == collect(1:50)

# full buffer
push!(b, d)
push!(b, d)
@test get_last_N_indices(b, 10) == collect(41:50)
@test get_last_N_indices(b, 1) == [50]
@test get_last_N_indices(b, 50) == collect(1:50)
@test get_last_N_indices(b, 51) == [100, collect(1:50)...]
@test get_last_N_indices(b, 100) == [collect(51:100)..., collect(1:50)...]
@test get_last_N_indices(b, 1000) == [collect(51:100)..., collect(1:50)...]


## Priority Param Construction
p = PriorityParams(100, α=0.7)
@test length(p.priorities) == 100
@test p.α == 0.7f0
@test p.max_priority == 1.0f0

p2 = PriorityParams(1000, p)
@test length(p2.priorities)==1000
@test p2.α == p.α
@test p2.max_priority == p.max_priority

## Construction
b = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 100,)
bpriority = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 50, prioritized=true)
USE_CUDA && (b_gpu = b |> gpu)

bcopy = ExperienceBuffer(deepcopy(b.data), elements=0)
@test b.data == bcopy.data
@test b.priority_params == bcopy.priority_params
bcopy = ExperienceBuffer(deepcopy(b.data), prioritized=true)
@test isprioritized(bcopy)

@test b isa ExperienceBuffer{Array}
if USE_CUDA 
    @test b_gpu isa ExperienceBuffer{CuArray}
    @test length(keys(b.data)) == length(keys(b_gpu.data))
end

@test :s in keys(b.data)
@test :a in keys(b.data)
@test :sp in keys(b.data)
@test :r in keys(b.data)
@test :done in keys(b.data) 

@test !haskey(b, :weight)
@test haskey(bpriority, :weight)
@test bpriority.data[:weight] == ones(Float32, 1, 50)
@test size(b[:a]) == (4,0)

# Buffer_like function
bsmall = buffer_like(b, capacity=3, device=cpu)
@test Crux.device(bsmall) == cpu
@test keys(bsmall) == keys(b)
@test capacity(bsmall) == 3
@test length(bsmall) == 0


## Base functions 
@test keys(b) == keys(b.data)

@test size(b[:s]) == (2,0)
@test length(b) == 0
@test capacity(b) == 100

USE_CUDA && @test size(b_gpu[:s]) == (2,0)
USE_CUDA && @test length(b_gpu) == 0
USE_CUDA && @test capacity(b_gpu) == 100

@test isprioritized(bpriority)
@test Crux.device(b) == cpu
@test Crux.device(bpriority) == cpu
USE_CUDA && @test Crux.device(b_gpu) == gpu

# Convert to and from GPU
USE_CUDA && @test Crux.device(gpu(b)) == gpu
USE_CUDA && @test Crux.device(cpu(b_gpu)) == cpu

## push!
#push dictionary with one element
d = Dict(:s => 2*ones(2,1), :a => ones(Bool, 4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1), :weight=>zeros(1,1))
push!(b, d)
@test length(b) == 1
@test b[:s] == 2*ones(2,1)
@test b[:a] == ones(Int, 4,1)
@test b[:sp] == ones(2,1)
@test b[:r] == ones(1,1)
@test b[:done] == zeros(1,1)

# push dictionary with more than one element
d = Dict(:s => 3*ones(2,3), :a => rand(4,3) .< 0.5, :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3), :weight=>zeros(1,3))
push!(b, d)
@test length(b) == 4
@test b[:s][:,2:end] ==  3*ones(2,3)
@test b[:a][:,2:end] == d[:a]
@test b[:sp][:,2:end] == 5*ones(2,3)
@test b[:r][:,2:end] == 6*ones(1,3)
@test b[:done][:,2:end] == ones(1,3)

# push a buffer
push!(b, b)
@test length(b) == 8
for k in keys(b)
    @test b[k][:, 1:4] == b[k][:, 5:8]
end


## Reservoir storage
b = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10,)
d3 = Dict(:s => 3*ones(2,3), :a => rand(4,3) .< 0.5, :sp => 5*ones(2,3), :r => 6*ones(1,3), :done => ones(1,3), :weight=>zeros(1,3), :episode_end=> ones(1,3))
d1 = Dict(:s => 2*ones(2,1), :a => ones(Bool, 4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1), :weight=>zeros(1,1), :episode_end => zeros(1,1))

for i=1:10
    push_reservoir!(b, d1)
    @test length(b) == i
end

push_reservoir!(b, d3)


## clear!
bcopy = deepcopy(b)
clear!(bcopy)
@test length(bcopy) == 0
@test bcopy.next_ind == 1

## minibatch
I = [1,2,4]
d2 = minibatch(b, I)
for k in keys(d2)
    @test all(d2[k] .== b[k][:, I])
end 

## Split
@test Crux.split_batches(100, [0.5, 0.5]) == [50,50]
@test Crux.split_batches(100, [1.0]) == [100]
@test Crux.split_batches(100, [1/3, 1/3, 1/3]) == [34,33,33]
@test_throws AssertionError Crux.split_batches(100, 0.4)

v = rand(3,100)
bsplit = ExperienceBuffer(Dict(:a => v))

b1, b2 = split(bsplit, [0.5, 0.5])
@test length(b1) == 50
@test b1[:a] == v[:,1:50]
@test length(b2) == 50
@test b2[:a] == v[:,51:100]


## update_priorities!
update_priorities!(bpriority, [1,2,3], [1., 2., 3.])
@test bpriority.priority_params.max_priority == 3.0
@test bpriority.priority_params.priorities[1] ≈ 1f0^bpriority.priority_params.α
@test bpriority.priority_params.priorities[2] ≈ 2f0^bpriority.priority_params.α
@test bpriority.priority_params.priorities[3] ≈ 3f0^bpriority.priority_params.α


push!(bpriority, d)
push!(bpriority, d)
@test bpriority.priority_params.max_priority == 3.0
for i=1:6
    @test bpriority.priority_params.priorities[i] ≈ 3f0^bpriority.priority_params.α
end
    

## sampling

# uniform sample
t = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
rand!(t, b)

t = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 3)
Random.seed!(0)
ids = rand(1:length(b), 3)
Random.seed!(0)
rand!(t, b)

for k in keys(t)
    @test t[k] == b[k][:,ids]
end

# Test the multi-buffer sampling
t1 = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
d = Dict(:s => ones(2,1), :a => ones(Bool, 4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t1, d)

t2 = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
d = Dict(:s => 2*ones(2,1), :a => ones(Bool, 4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t2, d)

t3 = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
d = Dict(:s => 3*ones(2,1), :a => ones(4,1), :sp => ones(2,1), :r => ones(1,1), :done => zeros(1,1))
push!(t3, d)

t = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
rand!(t, t1, t2, t3)

@test all(t[:s][:, 1:4] .== 1.0)
@test all(t[:s][:, 5:7] .== 2.0)
@test all(t[:s][:, 8:10] .== 3.0)



# Priority samples
bpriority[:s] .= rand(Float32, 2, 6)
update_priorities!(bpriority, [1:6...], [1.:6. ...])
t = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 1000, [:weight])
priorities = [bpriority.priority_params.priorities[i] for i=1:length(bpriority)]


rand!(t, bpriority)

# Test the the frequency of samples is proportional to their priority
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
b2d = ExperienceBuffer(ContinuousSpace((2,2), UInt8), DiscreteSpace(4), 100;)
@test ndims(b2d[:s]) == 3
b2d[:s]

d = Dict(:s => 3*ones(2,2,1), :a => ones(4,1), :sp => ones(2,2,1), :r => ones(1,1), :done => zeros(1,1))
push!(b2d, d)

@test all(b2d[:s] .== 3)

