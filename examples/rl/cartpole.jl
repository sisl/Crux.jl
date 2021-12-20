using POMDPs, Crux, Flux, POMDPGym

## Cartpole - V0
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)

A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))

# Solve with REINFORCE (~2 seconds)
𝒮_reinforce = REINFORCE(π=A(), S=S, N=10000, ΔN=500, a_opt=(epochs=5,), interaction_storage=[])
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C (~8 seconds)
𝒮_a2c = A2C(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
@time π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO (~15 seconds)
𝒮_ppo = PPO(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
@time π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN (~12 seconds)
𝒮_dqn = DQN(π=A(), S=S, N=10000, interaction_storage=[])
@time π_dqn = solve(𝒮_dqn, mdp)

# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn], title = "CartPole-V0 Training Curves", labels = ["REINFORCE", "A2C", "PPO", "DQN"])

# Produce a gif with the final policy
gif(mdp, π_ppo, "cartpole_policy.gif", max_steps=100)

## Optional - Save data for imitation learning
# using BSON
# s = Sampler(mdp, 𝒮_dqn.π, max_steps=100, required_columns=[:t])
# 
# data = steps!(s, Nsteps=10000)
# sum(data[:r])/100
# data[:expert_val] = ones(Float32, 1, 10000)
# data[:a]
# 
# data = ExperienceBuffer(data)
# BSON.@save "examples/il/expert_data/cartpole.bson" data

