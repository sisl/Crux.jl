using Crux, Flux, POMDPGym, Random, POMDPs

expert_buffer_size = 1000

## Cartpole - V0 (For DQN-GAIL)
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)

Q() = DiscreteNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
D_PG() = DiscreteNetwork(Chain(Dense(dim(S)[1] + length(as), 64, relu), Dense(64, 64, relu), Dense(64, 1, sigmoid)), as)
D_DQN() = DiscreteNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as), sigmoid)), as)
V() = ContinuousNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = DiscreteNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax), as)

# Solve with DQN
𝒮_dqn = DQNSolver(π = Q(), S = S, N=200)
π_dqn = solve(𝒮_dqn, mdp)

# Fill a buffer with expert trajectories
expert_trajectories = ExperienceBuffer(steps!(Sampler(mdp, π_dqn, S), Nsteps = expert_buffer_size))
sum(expert_trajectories[:r])


# Solve with DQN-GAIL
𝒮_gail = GAILSolver(D = D_DQN(), 
                    G = DQNSolver(π = Q(), S = S, N=000),
                    expert_buffer = expert_trajectories)
solve(𝒮_gail, mdp)

# Solve with PPO-GAIL
𝒮_ppo = PGSolver(π = ActorCritic(A(), V()), 
                S = S, N=300, ΔN = 500, loss = ppo())
𝒮_gail = GAILSolver(D = D_PG(), 
                    G = 𝒮_ppo,
                    expert_buffer = expert_trajectories)
solve(𝒮_gail, mdp)

