using Shard
using Test
using Flux
using POMDPs

## DQN Policy
π = DQNPolicy(Chain(Dense(2,32, relu), Dense(32, 4)), (1,2,3,4); device = cpu)
π_gpu = DQNPolicy(Chain(Dense(2,32, relu), Dense(32, 4)), (1,2,3,4); device = gpu)

@test network(π, cpu) == [π.Q]
@test network(π, gpu) == [nothing]
@test network(π_gpu, cpu) == [π_gpu.Q]
@test network(π_gpu, gpu) == [π_gpu.Q_GPU]

s = rand(2)
@test action(π, s) == argmax(π.Q(s))
@test action(π_gpu, s) == argmax(π_gpu.Q(s))

@test π.Q(s) == π.Q⁻(s)

sb = rand(2,100)
@test size(value(π, sb)) == (4,100)
@test value(π, sb) == π.Q(sb)
@test size(value(π_gpu, sb)) == (4,100)
@test mean(abs.(value(π_gpu, sb) .- (value(π_gpu, sb |> gpu) |> cpu))) < 1e-7

## sync!
π_gpu.Q_GPU = π.Q |> gpu
@test all(value(π_gpu, sb |> gpu) |> cpu .≈ value(π, sb))

