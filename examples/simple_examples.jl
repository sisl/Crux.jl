using POMDPs, Crux, Flux, POMDPGym

## GridWorld
mdp = GridWorldMDP()
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
Q() = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)))
A() = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
V() = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 32, relu), Dense(32, 32, relu), Dense(32, 1))

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q = Q(), actions = as), S = S, N=100000)
policy_dqn = solve(𝒮_dqn, mdp)

# Solve with vanilla policy gradient
𝒮_vpg = VPGSolver(π = CategoricalPolicy(A = A(), actions = as), S = S,  N=100000, baseline = Baseline(V = V()), batch_size = 128,)
policy_vpg = solve(𝒮_vpg, mdp)

# Plot the learning curve
p = plot_learning([𝒮_dqn, 𝒮_vpg], title = "GridWorld Training Curves", labels = ["DQN", "VPG"])

# Produce a gif with the final policy
gif(mdp, policy_dqn, "dqn_gridworld.gif")
gif(mdp, policy_vpg, "vpg_gridworld.gif")


## LavaWorld
mdp = LavaWorldMDP()
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
Q() = Chain(x -> reshape(x, 105, :), Dense(105, 64, relu), Dense(64, 64, relu), Dense(64, length(as)))
A() = Chain(x -> reshape(x, 105, :), Dense(105, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
V() = Chain(x -> reshape(x, 105, :), Dense(105, 32, relu), Dense(32, 32, relu), Dense(32, 1))

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q = Q(), actions = as), S = S, N=100000)
policy_dqn = solve(𝒮_dqn, mdp)

# Solve with vanilla policy gradient
𝒮_vpg = VPGSolver(π = CategoricalPolicy(A = A(), actions = as), S = S,  N=100000, baseline = Baseline(V = V()), batch_size = 128,)
policy_vpg = solve(𝒮_vpg, mdp)

# Plot the learning curve
p = plot_learning([𝒮_dqn, 𝒮_vpg], title = "LavaWorld Training Curves", labels = ["DQN", "VPG"])

# Produce a gif with the final policy
gif(mdp, policy_dqn, "dqn_lavaworld.gif")
gif(mdp, policy_vpg, "vpg_lavaworld.gif")


## LavaWorld
mdp = PendulumMDP(actions = [-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
S = state_space(mdp)

# Define the networks we will use
Q() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), x -> 200f0.*x .- 200f0)
A() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as)), softmax)
V() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 32, relu), Dense(32, 32, relu), Dense(32, 1), x -> 200f0.*x .- 200f0)
μ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))

# Solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q = Q(), actions = as), S = S, N=100000)
policy_dqn = solve(𝒮_dqn, mdp)

# Solve with vanilla policy gradient
𝒮_vpg = VPGSolver(π = CategoricalPolicy(A = A(), actions = as), S = S,  N=100000, baseline = Baseline(V = V()), batch_size = 128, opt = ADAM(1e-4))
policy_vpg = solve(𝒮_vpg, mdp)

𝒮_vpg_cont = VPGSolver(π = GaussianPolicy(μ = μ(), logΣ = zeros(Float32, 1)), S = S,  
        N=1000000, baseline = Baseline(V = V()), batch_size = 1028, opt = ADAM(1e-3))
policy_vpg_cont = solve(𝒮_vpg_cont, mdp)
𝒮_vpg_cont.π.logΣ
𝒮_vpg_cont.π.μ(rand(initialstate(mdp)))

# Plot the learning curve
p = plot_learning([𝒮_dqn, 𝒮_vpg], title = "Pendulum Swingup Training Curves", labels = ["DQN", "VPG", "VPG - Continuous"])

# Produce a gif with the final policy
gif(mdp, policy_dqn, "dqn_pendulum.gif", max_steps = 200)
gif(mdp, policy_vpg, "vpg_pendulum.gif", max_steps = 200)
gif(mdp, policy_vpg_cont, "vpg_cont_pendulum.gif", max_steps = 200)

