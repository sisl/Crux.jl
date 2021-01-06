using POMDPs, Crux, Flux, POMDPGym

## Cartpole - V0
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)

Q() = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)))
V() = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1))
A() = Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)

# Solve with REINFORCE
𝒮_reinforce = PGSolver(π = CategoricalPolicy(A = A(), actions = as),
                S = S, N=10000, ΔN = 500, loss = reinforce())
π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C
𝒮_a2c = PGSolver(π = ActorCritic(CategoricalPolicy(A = A(), actions = as), V()), 
                S = S, N=10000, ΔN = 500, loss = a2c())
π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO
𝒮_ppo = PGSolver(π = ActorCritic(CategoricalPolicy(A = A(), actions = as), V()), 
                S = S, N=10000, ΔN = 500, loss = ppo())
π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q = Q(), actions = as), S = S, N=10000)
π_dqn = solve(𝒮_dqn, mdp)

# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn], title = "CartPole-V0 Training Curves", labels = ["REINFORCE", "A2C", "PPO", "DQN"])

# Produce a gif with the final policy
gif(mdp, π_ppo, "cartpole_policy.gif")