include("../src/includes.jl")
using Test
using POMDPModels
using Flux
using LinearAlgebra

## sdim and adim
mdp = SimpleGridWorld()
@test sdim(mdp) == 2
@test adim(mdp) == 4

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

## Gpu stuff
vcpu = zeros(Float32, 10, 10)
@test device(vcpu) == cpu

vgpu = todevice(vcpu, gpu)
@test device(vgpu) == gpu

@test isnothing(todevice(vcpu, cpu))

c1 = Chain(Dense(5, 5, relu))
c2 = Chain(Dense(5, 5, relu))
c3 = Chain(Dense(5, 5, relu)) |> gpu

@test c1[1].W != c2[1].W
copyto!(c1, c2)

@test c1[1].W == c2[1].W

copyto!(c3, c2)
@test c3[1].W isa CuArray
@test cpu(c3[1].W) == c2[1].W


## Flux Stuff
W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Flux.params(W, b)
grads = Flux.gradient(() -> loss(x, y), θ)
@test norm(grads) > 2






