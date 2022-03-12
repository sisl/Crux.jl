using Crux, Flux
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

