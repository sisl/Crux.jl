using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
DDPG_Q() = Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1),x -> 200f0*x .- 200f0)
Q() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), x -> 200f0*x .- 200f0)
A() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
μ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))
V() = DeterministicNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1), x -> 200f0*x .- 200f0), 1)

early_stopping = (info) -> info[:kl] > 0.015

# Solve with REINFORCE
𝒮_reinforce = PGSolver(π = GaussianPolicy(μ(), zeros(Float32, 1)),
                S = S, N=100000, ΔN = 2048, loss = reinforce(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-4)), batch_size = 512, epochs = 100, early_stopping = early_stopping)
π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C
𝒮_a2c = PGSolver(π = ActorCritic(GaussianPolicy(μ(), zeros(Float32, 1)), V()), 
                S = S, N=100000, ΔN = 2048, loss = a2c(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-4)), batch_size = 512, epochs = 100,  early_stopping = early_stopping)
π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO
𝒮_ppo = PGSolver(π = ActorCritic(GaussianPolicy(μ(), zeros(Float32, 1)), V()), 
                S = S, N=100000, ΔN = 2048, loss = ppo(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-3)), batch_size = 512, epochs = 100,  early_stopping = early_stopping)
π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q(), as), S = S, N=100000)
π_dqn = solve(𝒮_dqn, mdp)

# Solve with DDPG
𝒮_ddpg = DDPGSolver(π = DDPGPolicy(μ(), DDPG_Q(), action_dim = 1), S = S, N=100000)
π_ddpg = solve(𝒮_ddpg, mdp)


# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_ddpg], title = "Pendulum Swingup Training Curves", labels = ["REINFORCE", "A2C", "PPO", "DQN", "DDPG"])

# Produce a gif with the final policy
gif(mdp, π_dqn, "pendulum.gif", max_steps = 200)

