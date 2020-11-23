include("../src/includes.jl")
using Test

vgpu = CuArray{Float32, 2}(undef, 10, 10)
@test device(vgpu) == gpu

vcpu = Array{Float32, 2}(undef, 10, 10)
@test device(vcpu) == cpu


## Binary heap test
v = rand(19)
b = MutableBinaryHeap{Float64, DataStructures.FasterForward}(v)
@test first(b) == minimum(v)

push!(b, 1.0)
update!(b, 1, 0.)
first(b)
b[2] .== v
b[2]

vlarge = rand(10000)
t = FenwickTree(vlarge)
h = MutableBinaryHeap{Float64, DataStructures.FasterForward}(vlarge)

@time first(h)
@time push!(h, 0.5)
@time h[999]

@time t[999]
@time inv_query(t, 200)
@time value(t, 999)

## FenwickTree
t = FenwickTree(ones(10))
v = rand(16)
t2pow = FenwickTree(v)

# Inverse query on non power of 2 arrays
@test inv_query(t, 0.5) == 1
@test inv_query(t, 1.0) == 1
@test inv_query(t, 0) == 1
@test inv_query(t, -1) == 1
@test inv_query(t, 3.0) == 3
@test inv_query(t, 3.5) == 4
@test inv_query(t, 4.0) == 4
@test inv_query(t, 9.5) == 10
@test inv_query(t, 10) == 10
@test inv_query(t, 11) == 11

# power of 2 arrays
for i=1:1000
    r = 5*rand()
    ind = inv_query(t2pow, r)

    @test t2pow[ind] ≥ r
    @test t2pow[ind-1] < r
end

@test inv_query(t2pow, t2pow[16]) == 16