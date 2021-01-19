# Note: This setup should get at least the performance of the openai-spinning up benchmarks
using Crux, Flux, POMDPs, POMDPGym, Distributions
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
mdp = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])

# Initializations that match the default PyTorch initializations
Winit(out, in) = rand(Uniform(Float32(-sqrt(1/in)), Float32(sqrt(1/in))), out, in)
binit(in) = (out) -> rand(Uniform(Float32(-sqrt(1/in)), Float32(sqrt(1/in))), out)

# Build the networks
μ() = Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), Dense(64, 32, tanh, initW = Winit, initb = binit(64)), Dense(32, adim, initW = Winit, initb = binit(32)))
V() = Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), Dense(64, 32, tanh, initW = Winit, initb = binit(64)), Dense(32, 1, initW = Winit, initb = binit(32)))
log_std() = -0.5*ones(Float32, adim)

# Solve with ppo
𝒮_ppo = PGSolver(π = ActorCritic(GaussianPolicy(μ(), log_std()), V()), 
                 S = S, 
                 max_steps = 1000, 
                 loss = ppo(λₑ = 0f0),
                 ΔN = 4000,
                 λ_gae = 0.97,
                 batch_size = 4000,
                 epochs = 80,
                 early_stopping = (info) -> info[:kl] > 0.015,
                 N = 3000000, 
                 opt = ADAM(3e-4),
                 opt_v = ADAM(1e-3)
                 )
solve(𝒮_ppo, mdp)

# Plot the learning curve
p = plot_learning(𝒮_ppo, title = "HalfCheetah Training Curves")

# Produce a gif with the final policy
gif(mdp, 𝒮_ppo.π, "mujoco.gif")

