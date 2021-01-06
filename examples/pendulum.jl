using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
Q() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), x -> 200f0.*x .- 200f0)
A() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
μ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))
V() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1), x -> 200f0.*x .- 200f0)

# Solve with REINFORCE
𝒮_reinforce = PGSolver(π = GaussianPolicy(μ = μ(), logΣ = zeros(Float32, 1)),
                S = S, N=100000, ΔN = 2048, loss = reinforce(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-4)), batch_size = 512, epochs = 10)
π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C
𝒮_a2c = PGSolver(π = ActorCritic(GaussianPolicy(μ = μ(), logΣ = zeros(Float32, 1)), V()), 
                S = S, N=100000, ΔN = 2048, loss = a2c(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-4)), batch_size = 512, epochs = 10)
π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO
𝒮_ppo = PGSolver(π = ActorCritic(GaussianPolicy(μ = μ(), logΣ = zeros(Float32, 1)), V()), 
                S = S, N=100000, ΔN = 2048, loss = ppo(), opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-3)), batch_size = 512, epochs = 100)
π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q = Q(), actions = as), S = S, N=100000)
π_dqn = solve(𝒮_dqn, mdp)


# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn], title = "Pendulum Swingup Training Curves", labels = ["REINFORCE", "A2C", "PPO", "DQN"])

# Produce a gif with the final policy
gif(mdp, π_dqn, "pendulum.gif", max_steps = 200)
