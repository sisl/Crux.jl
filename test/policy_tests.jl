using Crux
using Test
using Flux
using POMDPs

## DQN Policy
π = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)), actions = [1,2,3,4])
π_gpu = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)) |> gpu, actions = [1,2,3,4])

s = rand(2)
@test action(π, s) == argmax(π.Q(s))
@test action(π_gpu, s) == argmax(mdcall(π_gpu.Q,s, π_gpu.device))

@test π.Q(s) == π.Q⁻(s)

sb = rand(2,100)
@test size(value(π, sb)) == (4,100)
@test value(π, sb) == π.Q(sb)
@test size(value(π_gpu, sb)) == (4,100)
@test mean(abs.(value(π_gpu, sb) .- (value(π_gpu, sb |> gpu) |> cpu))) < 1e-7

