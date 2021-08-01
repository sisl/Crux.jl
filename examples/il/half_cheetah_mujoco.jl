using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies, BSON
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
mdp = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

## (Load from bson) get expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/half_cheetah_mujoco.bson")[:data]
μ_s = mean(expert_trajectories[:s], dims=2)[:]
σ_s = std(expert_trajectories[:s], dims=2)[:] .+ 1f-3
S = state_space(mdp, μ=μ_s, σ=σ_s)

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(in) = (out) -> Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

## Behavioral Cloning 
μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh), Dense(64, 32, tanh), Dense(32, adim)))
log_std() = -0.5f0*ones(Float32, adim)

𝒮_bc = BC(π=μ(), 
          𝒟_demo=expert_trajectories, 
          S=S,
          opt=(epochs=100000, batch_size=1024), 
          log=(period=500,),
          max_steps=1000)
solve(𝒮_bc, mdp)


## AdVIL 
A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 6, init=Flux.orthogonal)))
D() = ContinuousNetwork(Chain(Dense(S.dims[1] + adim, 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 1, init=Flux.orthogonal)))
            
𝒮_advil = AdVIL(π=ActorCritic(A(),D()), 𝒟_demo=expert_trajectories, S=S, a_opt=(epochs=100000, optimizer=ADAM(8f-6), batch_size=1024), c_opt=(optimizer=ADAM(8e-4),), max_steps=1000)
solve(𝒮_advil, mdp)

