using Crux, Flux, POMDPGym, Random, POMDPs, BSON

## Cartpole
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)

Disc() = ContinuousNetwork(Chain(Dense(6, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
V() = ContinuousNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

# Fill a buffer with expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/cartpole.bson")[:data]

# Solve with PPO-GAIL
γ = Float32(discount(mdp))
𝒮_gail = OnPolicyGAIL(D=Disc(), γ=γ, gan_loss=GAN_BCELoss(), 𝒟_demo=expert_trajectories, solver=PPO, π=ActorCritic(A(), V()), S=S, N=40000, ΔN=1024, d_opt=(batch_size=1024, epochs=80))
solve(𝒮_gail, mdp)

# Solve with Behavioral Cloning
𝒮_bc = BC(π=A(), 𝒟_demo=expert_trajectories, S=S, opt=(epochs=100,), log=(period=10,))
N = solve(𝒮_bc, mdp)

# Solve with IQ-Learn
