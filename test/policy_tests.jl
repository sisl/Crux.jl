using Crux
using Test
using Flux
using POMDPs
using POMDPGym
using POMDPPolicies


## DQN Policy
π_dqn = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)), actions = [1,2,3,4])
π_gpu = DQNPolicy(Q = Chain(Dense(2,32, relu), Dense(32, 4)) |> gpu, actions = [1,2,3,4])

s = rand(2)
@test action(π_dqn, s) == argmax(π_dqn.Q(s))
@test action(π_gpu, s) == argmax(mdcall(π_gpu.Q,s, π_gpu.device))

@test π_dqn.Q(s) == π_dqn.Q⁻(s)

sb = rand(2,100)
@test size(value(π_dqn, sb)) == (4,100)
@test value(π_dqn, sb) == π_dqn.Q(sb)
@test size(value(π_gpu, sb)) == (4,100)
@test mean(abs.(value(π_gpu, sb) .- (value(π_gpu, sb |> gpu) |> cpu))) < 1e-7

