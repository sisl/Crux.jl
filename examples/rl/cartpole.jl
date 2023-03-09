using POMDPs, Crux, Flux, POMDPGym

## Cartpole - V0
mdp = GymPOMDP(:CartPole, version = :v1)
as = actions(mdp)
S = state_space(mdp)

A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
SoftA(α::Float32) = SoftDiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as;α=α)

# collection and c_opt_epoch optimization
ΔNs=[1,2,4]
epochs = [1,5,10,50]
mix = Iterators.product(ΔNs,epochs)  
𝒮_sqls_2 = [SoftQ(π=SoftA(Float32(0.5)), S=S, N=10000, 
    ΔN=dn, c_opt=(;epochs=e), interaction_storage=[]) for (dn,e) in mix]
π_sqls_2 = [@time solve(x, mdp) for x in 𝒮_sqls_2]
p = plot_learning(𝒮_sqls_2, title = "CartPole-V0 SoftQ Tradeoff Curves", 
    labels = ["SQL ΔN=($dn),ep=($e)" for (dn,e) in mix])
Crux.savefig(p, "examples/rl/cartpole_soft_q_tradeoffs.pdf")


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

# Solve with SoftQLearning w/ varying α (~12 seconds)
αs = Vector{Float32}([1,0.5,0.2,0.1])
𝒮_sqls = [SoftQ(π=SoftA(α), S=S, N=10000, interaction_storage=[]) for α in αs]
π_sqls = [@time solve(𝒮_sqls[i], mdp) for i=1:length(αs)]

# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_sqls...], title = "CartPole-V0 Training Curves", 
    labels = ["REINFORCE", "A2C", "PPO", "DQN", ["SQL ($i)" for i in αs]...])
Crux.savefig(p, "examples/rl/cartpole_training.pdf")

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

