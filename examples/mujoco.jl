# Note: This setup should get at least the performance of the openai-spinning up benchmarks
using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
mdp = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(in) = (out) -> Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Build the networks
μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), 
                              Dense(64, 32, tanh, initW = Winit, initb = binit(64)), 
                              Dense(32, adim, initW = Winit, initb = binit(32))))
V() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), 
                              Dense(64, 32, initW = Winit, initb = binit(64)), 
                             Dense(32, 1, initW = Winit, initb = binit(32))))
log_std() = -0.5f0*ones(Float32, adim)

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
                 opt_critic = ADAM(1e-3)
                 )
solve(𝒮_ppo, mdp)


idim = S.dims[1] + adim
Q() = ContinuousNetwork(Chain(Dense(idim, 256, relu, initW = Winit, initb = binit(idim)), 
            Dense(256, 256, relu, initW = Winit, initb = binit(256)), 
            Dense(256, 1, initW = Winit, initb = binit(256))))
μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, initW = Winit, initb = binit(idim)), 
            Dense(256, 256, relu, initW = Winit, initb = binit(256)), 
            Dense(256, 6, tanh, initW = Winit, initb = binit(256))))



# Solve with DDPG
𝒮_ddpg = DDPGSolver( π = ActorCritic(μ(), Q()),
                     S = S, 
                     N = 3000000, 
                     max_steps = 1000,
                     ΔN = 50,
                     buffer_size = 1000000,
                     buffer_init = 1000,
                     τ = 0.005f0,
                     batch_size = 100,
                     opt_actor = ADAM(1f-3),
                     opt_critic = ADAM(1f-3),
                     exploration_policy = FirstExplorePolicy(10000, rand_policy, GaussianNoiseExplorationPolicy(0.1f0, clip_min = amin, clip_max = amax)),
                     )
                     
solve(𝒮_ddpg, mdp)

# Plot the learning curve
p = plot_learning([𝒮_ppo, 𝒮_ddpg], title = "HalfCheetah Training Curves", labels = ["PPO, DDPG"])

# Produce a gif with the final policy
gif(mdp, 𝒮_ddpg.π, "mujoco.gif")

