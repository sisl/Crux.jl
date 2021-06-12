using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies, BSON
init_mujoco_render() # Required for visualization

# Construct the PyBullet environment
using PyCall
pyimport("pybullet_envs")
mdp = GymPOMDP(:HalfCheetahBulletEnv)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

## (Load from bson) get expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/half_cheetah_pybullet.bson")[:data]
μ_s = mean(expert_trajectories[:s], dims=2)[:]
σ_s = std(expert_trajectories[:s], dims=2)[:] .+ 1f-3
S = state_space(mdp, μ=μ_s, σ=σ_s)


# a1 = expert_trajectories[:a]
# 
# expert_perf = sum(expert_trajectories.data[:r]) / length(episodes(expert_trajectories))

# (Load from csv) get expert trajectories
# using CSV, DataFrames
# a = Float32.(Matrix(CSV.read("examples/il/expert_data/half_cheetah_acts.csv", DataFrame, header=0))')
# s = Float32.(Matrix(CSV.read("examples/il/expert_data/half_cheetah_obs.csv", DataFrame, header=0))')
# expert_trajectories = ExperienceBuffer(Dict(:s=>s, :a=>a))
# μ_s = Float32[-0.28287876, 0., 1., 0.51424587, 0., -0.00496594, 0., 0.7428437, -0.8199211, -0.00579686, -0.9570886, 0.00579302, 0.28927535, -0.01396482, 0.95324045, 0.00380081, 0.5716587, 0.00654798, 0.4403377, -0.00796789, 0.8184, 0., 0., 0.719, 0., 0.]
# σ_s = Float32[0.03372439, 0.001, 0.001, 0.10826081, 0.001, 0.07804839, 0.001, 0.14742103, 0.2113209, 0.25461975, 0.22941712, 0.17325062, 0.29873753, 0.47041118, 0.1173827, 0.14787048, 0.09817489, 0.27447778, 0.28768054, 0.75099736, 0.3865124, 0.001, 0.001, 0.45049468, 0.001, 0.001]

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(in) = (out) -> Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))


## Setup params

# Solve with PPO-GAIL
# 𝒮_gail = GAIL(;D=D(), 
#               gan_loss=GAN_LSLoss(), 
#               𝒟_demo=expert_trajectories, 
#               solver=PPO, 
#               π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), 
#               ΔN=4000, 
#               a_opt=(batch_size=4000, epochs=80, optimizer=ADAM(3e-4)),
#               shared...)
# solve(𝒮_gail, mdp)


μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh), Dense(64, 32, tanh), Dense(32, adim)))
log_std() = -0.5f0*ones(Float32, adim)

𝒮_bc = BC(π=GaussianPolicy(μ(), log_std()), 
          𝒟_demo=expert_trajectories, 
          S=S, 
          # window=500,
          opt=(epochs=100000, early_stopping=(args...)->false,), 
          log=(period=100,),
          max_steps=1000)
solve(𝒮_bc, mdp)



A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 6, init=Flux.orthogonal)))
D() = ContinuousNetwork(Chain(Dense(S.dims[1] + adim, 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 1, init=Flux.orthogonal)))
            
𝒮_advil = AdVIL(π=ActorCritic(A(),D()), 𝒟_demo=expert_trajectories, S=S, a_opt=(epochs=100000, optimizer=ADAM(8f-6), batch_size=1024), c_opt=(optimizer=ADAM(8e-4),), max_steps=1000, log=(period=100,))
solve(𝒮_advil, mdp)

