using Crux
using Test
using POMDPModels
using Flux
using LinearAlgebra
using CUDA

## Spaces
mdp = SimpleGridWorld()

s1 = DiscreteSpace(5)
@test s1 isa AbstractSpace
@test s1 isa DiscreteSpace
@test s1.N == 5
@test type(s1) == Bool
@test Crux.dim(s1) == (5,)
@test s1.vals == [1,2,3,4,5]
@test tovec(4, s1) == Bool[0,0,0,1,0]

s2 = DiscreteSpace(5, [:a, :b, :c, :d, :e])
@test s2 isa AbstractSpace
@test s2 isa DiscreteSpace
@test s2.N == 5
@test type(s2) == Bool
@test Crux.dim(s2) == (5,)
@test s2.vals == [:a, :b, :c, :d, :e]
@test tovec(:d, s2) == Bool[0,0,0,1,0]

s3 = ContinuousSpace((3,4), Float64)
@test s3 isa AbstractSpace
@test s3 isa ContinuousSpace
@test s3.dims == (3,4)
@test type(s3) == Float64
@test Crux.dim(s3) == (3,4)
@test tovec(zeros(3,4), s3) == zeros(3, 4)

s3 = ContinuousSpace((3,4), Float64, 1f0, 2f0)
@test s3 isa AbstractSpace
@test s3 isa ContinuousSpace
@test s3.dims == (3,4)
@test type(s3) == Float64
@test Crux.dim(s3) == (3,4)
@test tovec(zeros(3,4),s3) == -0.5*ones(3, 4)

s4 = state_space(mdp)
@test s4 isa ContinuousSpace
@test Crux.dim(s4) == (2,)

## Gpu stuff
vcpu = zeros(Float32, 10, 10)
vgpu = cu(zeros(Float32, 10, 10))
@test Crux.device(vcpu) == cpu
@test Crux.device(vgpu) == gpu
@test Crux.device(view(vcpu,:,1)) == cpu
@test Crux.device(view(vgpu,:,1)) == gpu

c_cpu = Chain(Dense(5,2))
c_gpu = Chain(Dense(5,2)) |> gpu
@test Crux.device(c_cpu) == cpu
@test Crux.device(c_gpu) == gpu

@test Crux.device(mdcall(c_cpu, rand(5), cpu)) == cpu
@test Crux.device(mdcall(c_gpu, rand(5), gpu)) == cpu
@test Crux.device(mdcall(c_cpu, cu(rand(5)), cpu)) == gpu
@test Crux.device(mdcall(c_gpu, cu(rand(5)), gpu)) == gpu

v = zeros(4,4,4)
@test size(bslice(v, 2)) == (4,4)

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






