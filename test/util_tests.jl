using Crux
using Test
using POMDPModels
using Flux
using LinearAlgebra
using CUDA
using Distributions

# bslice
v = zeros(4,4,4)
@test size(bslice(v, 2)) == (4,4)

# Constant Layer
c1 = ConstantLayer(ones(10))
@test Crux.device(c1) == cpu
@test c1(rand(100)) == c1.vec
@test Flux.params(c1)[1] == c1.vec

if USE_CUDA
    c2 = c1 |> gpu
    @test Crux.device(c2) == gpu
    @test c2(rand(100)) == c2.vec
end

# Distribution stuff
objs = [:up, :down]
o = ObjectCategorical(objs)
@test o isa DiscreteUnivariateDistribution
@test o.objs == objs
@test o.cat.p == Categorical(2).p


@test rand(o) in objs
@test size(rand(o, 10)) == (10,)

@test logpdf(o, [:up]) == logpdf(o, [:down])
@test size(logpdf(o, rand(o,10))) == (1,10)

## Flux Stuff
W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Flux.params(W, b)
grads = Flux.gradient(() -> loss(x, y), θ)

@test norm(grads) > 1


##  MultitaskDecay Schedule
m = MultitaskDecaySchedule(10, [1,2,3])
l = Crux.LinearDecaySchedule(1.0, 0.1, 10)

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-20)
end

m = MultitaskDecaySchedule(10, [1,2,1])

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-10)
end

@test m(31) == 0.1
@test m(0) == 1





