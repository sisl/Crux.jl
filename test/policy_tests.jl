include("../src/includes.jl")
using Test

Qnet = Chain(Dense(2,32, relu), Dense(32, 4))
π = DQNPolicy(Qnet, mdp; device = cpu)

@test size(value(π, b[:s])) == (4,100)

y = target(Qnet, b, 0.9)
sum(value(π, b[:s]) .* b[:a], dims = 1)