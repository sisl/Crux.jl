using POMDPs, POMDPGym, Test, Crux, Flux, Random, BSON
function test_solver(𝒮fn, mdp, π...)
    # run it once
    Random.seed!(0)
    π1 = solve(𝒮fn(deepcopy.(π)...), deepcopy(mdp))
    
    # run it again on the cpu with the same rng
    Random.seed!(0)
    π2 = solve(𝒮fn(deepcopy.(π)...), deepcopy(mdp))
    
    # Run it on the gpu
    Random.seed!(0)
    π3 = solve(𝒮fn(gpu.(deepcopy.(π))...), deepcopy(mdp))
    
    # compare the results
    s = rand(Crux.dim(state_space(mdp))...)
    try
        value(π[1], s)
        @test all(value(π1, s) .≈ value(π2, s))
        @test all(abs.(value(π2, s) .- value(π3, s)) .< 1e-3)
    catch
        @test all(action(π1, s) .≈ action(π2, s))
        @test all(abs.(action(π2, s) .- action(π3, s)) .< 1e-3)
    end
end

## Training params
N = 100
ΔN = 50

## discrete RL
discrete_mdp = GridWorldMDP()
S = state_space(discrete_mdp)
A() = DiscreteNetwork(Chain(Dense(2, 32, relu), Dense(32, 4)), actions(discrete_mdp))
V() = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1)))
AC() = ActorCritic(A(), V())

test_solver((π) -> REINFORCE(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, A())
test_solver((π) -> A2C(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, AC())
test_solver((π) -> PPO(π=π, S=S, N=N, ΔN=ΔN), discrete_mdp, AC())
test_solver((π) -> DQN(π=π, S=S, N=N), discrete_mdp, A())


## Continuous RL 
continuous_mdp = PendulumMDP()
S = state_space(continuous_mdp)
QSA() = ContinuousNetwork(Chain(Dense(3, 32, tanh), Dense(32, 1)))
V() = ContinuousNetwork(Chain( Dense(2, 32, relu), Dense(32, 1)))
A() = ContinuousNetwork(Chain(Dense(2, 32, relu), Dense(32, 1, tanh)), 1)
G() = GaussianPolicy(A(), zeros(Float32, 1))

test_solver((π) -> REINFORCE(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, G())
test_solver((π) -> A2C(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), V()))
test_solver((π) -> PPO(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), V()))
test_solver((π) -> DDPG(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(A(), QSA()))
test_solver((π) -> TD3(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(A(), DoubleNetwork(QSA(), QSA())))
test_solver((π) -> SAC(π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), DoubleNetwork(QSA(), QSA())))


# Continuous IL
𝒟_expert = expert_trajectories = BSON.load("examples/il/expert_data/pendulum.bson")[:data]
D() = ContinuousNetwork(Chain(DenseSN(3, 32, relu), DenseSN(32, 1)))

test_solver((π, D) -> GAIL(D=D, 𝒟_expert=𝒟_expert, π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), V()), QSA())
test_solver(π -> BC(π=π, 𝒟_expert=𝒟_expert, S=S, opt=(epochs=1,)), continuous_mdp, A())
# NOTE: gradient penalty on the gpu only plays nicely with tanh, not relus in the discriminator?
test_solver((π) -> AdVIL(𝒟_expert=𝒟_expert, π=π, S=S, a_opt=(epochs=1,) ), continuous_mdp, ActorCritic(A(), QSA()))
test_solver((π) -> ValueDICE(𝒟_expert=𝒟_expert, π=π, S=S, N=N, ΔN=ΔN), continuous_mdp, ActorCritic(G(), QSA()))

