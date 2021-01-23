using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
QS() = DiscreteNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)

G() = GaussianPolicy(A(), zeros(Float32, 1))


# Solve with REINFORCE
𝒮_reinforce = PGSolver(π=G(), S=S, N=100000, ΔN=2048, loss=reinforce(), batch_size=512)
π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C
𝒮_a2c = PGSolver(π=ActorCritic(G(), V()), S=S, N=100000, ΔN=2048, loss=a2c(), batch_size=512)
π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO
𝒮_ppo = PGSolver(π=ActorCritic(G(), V()), S=S, N=100000, ΔN=2048, loss=ppo(), batch_size=512)
π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN
𝒮_dqn = DQNSolver(π=QS(), S=S, N=100000)
π_dqn = solve(𝒮_dqn, mdp)

# Solve with DDPG
𝒮_ddpg = DDPGSolver(π=ActorCritic(A(), QSA()), S=S, N=100000)
π_ddpg = solve(𝒮_ddpg, mdp)


# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_ddpg], title="Pendulum Swingup Training Curves", labels=["REINFORCE", "A2C", "PPO", "DQN", "DDPG"])

# Produce a gif with the final policy
gif(mdp, π_dqn, "pendulum.gif", max_steps=200)

