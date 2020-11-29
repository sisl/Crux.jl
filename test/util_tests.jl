using Shard
using Test
using POMDPModels
using Flux
using LinearAlgebra

## sdim and adim
mdp = SimpleGridWorld()
@test sdim(mdp) == 2
@test adim(mdp) == 4

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






