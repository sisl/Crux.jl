using Crux, Flux, POMDPGym, Random, POMDPs, BSON

## Cartpole
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)
Crux.dim(S)[1] + 1

D() = ContinuousNetwork(Chain(DenseSN(Crux.dim(S)[1] + 1, 64, relu), DenseSN(64, 64, relu), DenseSN(64, 1)))
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)

# Fill a buffer with expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/cartpole.bson")[:data]

# Solve with PPO-GAIL
𝒮_gail = GAIL(D=D(), gan_loss = GAN_LSLoss(), 𝒟_expert=expert_trajectories, solver=PPO, π=ActorCritic(A(), V()), S=S, N=10000, ΔN=1000)
solve(𝒮_gail, mdp)

