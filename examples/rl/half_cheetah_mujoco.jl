# Note: This setup should get at least the performance of the openai-spinning up benchmarks
using Crux, Flux, POMDPs, POMDPGym, Distributions
import POMDPPolicies:FunctionPolicy
using Random
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
binit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Build the networks
idim = S.dims[1] + adim
# Networks for on-policy algorithms
μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, init=Winit, bias=binit(64, S.dims[1])), 
                              Dense(64, 32, tanh, init=Winit, bias=binit(32, 64)), 
                              Dense(32, adim, init=Winit, bias=binit(adim, 32))))
V() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, init=Winit, bias=binit(64, S.dims[1])), 
                              Dense(64, 32, init=Winit, bias=binit(32, 64)), 
                             Dense(32, 1, init=Winit, bias=binit(1, 32))))
log_std() = -0.5f0*ones(Float32, adim)

# Networks for off-policy algorithms
Q() = ContinuousNetwork(Chain(Dense(idim, 256, relu, init=Winit, bias=binit(256, idim)), 
            Dense(256, 256, relu, init=Winit, bias=binit(256, 256)), 
            Dense(256, 1, init=Winit, bias=binit(1,256)))) 
A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, init=Winit, bias=binit(256, S.dims[1])), 
            Dense(256, 256, relu, init=Winit, bias=binit(256, 256)), 
            Dense(256, 6, tanh, init=Winit, bias=binit(6, 256)))) 
function SAC_A()
    base = Chain(Dense(S.dims[1], 256, relu, init=Winit, bias=binit(256, S.dims[1])), 
                Dense(256, 256, relu, init=Winit, bias=binit(256, 256)))
    mu = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Winit, bias=binit(6, 256))))
    logΣ = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Winit, bias=binit(6, 256))))
    SquashedGaussianPolicy(mu, logΣ)
end

## Setup params
shared = (max_steps=1000, N=Int(3e6), S=S)
on_policy = (ΔN=4000, 
             λ_gae=0.97, 
             a_opt=(batch_size=4000, epochs=80, optimizer=ADAM(3e-4)), 
             c_opt=(batch_size=4000, epochs=80, optimizer=ADAM(1e-3)))
off_policy = (ΔN=50,
              max_steps=1000,
              log=(period=4000, fns=[log_undiscounted_return(3)]),
              buffer_size=Int(1e6), 
              buffer_init=1000, 
              c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=100, optimizer=ADAM(1e-3)), 
              π_explore=FirstExplorePolicy(10000, rand_policy, GaussianNoiseExplorationPolicy(0.1f0, a_min=amin, a_max=amax)))

## Run solvers 
𝒮_ppo = PPO(;π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), λe=0f0, shared..., on_policy...)
solve(𝒮_ppo, mdp)

# Solve with DDPG
𝒮_ddpg = DDPG(;π=ActorCritic(A(), Q()) |> gpu, shared..., off_policy...)  
solve(𝒮_ddpg, mdp)

# Solve with TD3
𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(Q(), Q())) |> gpu, shared..., off_policy..., 
                  π_smooth = GaussianNoiseExplorationPolicy(0.2f0, ϵ_min = -0.5f0, ϵ_max = 0.5f0, a_min = amin, a_max = amax))
solve(𝒮_td3, mdp)

# Solve with SAC
𝒮_sac = SAC(;π=ActorCritic(SAC_A(), DoubleNetwork(Q(), Q())) |> gpu, shared..., off_policy...)
solve(𝒮_sac, mdp)

# Plot the learning curve
p = plot_learning([𝒮_ppo, 𝒮_ddpg, 𝒮_td3, 𝒮_sac], title = "HalfCheetah Mujoco Training Curves", labels = ["PPO", "DDPG", "TD3", "SAC"])
Crux.savefig("examples/rl/half_cheetah_mujoco_benchmark.pdf")

# Produce a gif with the final policy
gif(mdp, 𝒮_ddpg.π, "half_cheetah_mujoco.gif")

## Save trajectories for imitation learning
# using BSON
# s = Sampler(mdp, 𝒮_ddpg.π, max_steps=1000, required_columns=[:t])
# 
# data = steps!(s, Nsteps=10000)
# sum(data[:r])/10
# data[:expert_val] = ones(Float32, 1, 10000)
# 
# data = ExperienceBuffer(data)
# BSON.@save "examples/il/expert_data/half_cheetah_mujoco.bson" data

