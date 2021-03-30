using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies, BSON
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
using PyCall
pyimport("pybullet_envs")
mdp = GymPOMDP(:HalfCheetahBulletEnv)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

# get expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/half_cheetah_pybullet.bson")[:data]

expert_perf = sum(expert_trajectories.data[:r]) / length(episodes(expert_trajectories))

s_mean = dropdims(mean(expert_trajectories[:s], dims=2), dims=2)
s_std = dropdims(std(expert_trajectories[:s], dims=2), dims=2)
s_std[s_std .== 0] .= 1
s_std

sa_mean = vcat(s_mean, zeros(Float32, 6))
sa_std = vcat(s_std, ones(Float32, 6))

function normalize(μ, σ²)
    (x) -> (x .- μ) ./ σ²
end    

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(in) = (out) -> Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Networks for on-policy algorithms
idim = S.dims[1] + adim

## Setup params

# Solve with PPO-GAIL
# 𝒮_gail = GAIL(;D=D(), 
#               gan_loss=GAN_LSLoss(), 
#               𝒟_expert=expert_trajectories, 
#               solver=PPO, 
#               π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), 
#               ΔN=4000, 
#               a_opt=(batch_size=4000, epochs=80, optimizer=ADAM(3e-4)),
#               shared...)
# solve(𝒮_gail, mdp)

# solve with valueDICE
# D() = ContinuousNetwork(Chain(normalize(sa_mean, sa_std), Dense(idim, 256, relu, init=Flux.orthogonal), 
#             Dense(256, 256, relu, init=Flux.orthogonal), 
#             Dense(256, 1, init=Flux.orthogonal, bias=false)))
# 
# function SAC_A()
#     base = Chain(normalize(s_mean, s_std), Dense(S.dims[1], 256, relu, init=Flux.orthogonal), 
#                 Dense(256, 256, relu, init=Flux.orthogonal))
#     mu = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Flux.orthogonal)))
#     logΣ = ContinuousNetwork(Chain(base..., Dense(256, 6, init=Flux.orthogonal)))
#     SquashedGaussianPolicy(mu, logΣ)
# end
# 
# 𝒮_valueDICE = ValueDICE(;π=ActorCritic(SAC_A(), D()),
#                         𝒟_expert=expert_trajectories, 
#                         max_steps=1000, 
#                         N=Int(1e5), 
#                         S=S,
#                         α=0.1,
#                         buffer_size=Int(1e6), 
#                         buffer_init=200,
#                         log=(period=1000,),
#                         c_opt=(batch_size=256, optimizer=ADAM(1e-5)), 
#                         a_opt=(batch_size=256, optimizer=ADAM(1e-5)))
# 
# solve(𝒮_valueDICE, mdp, mdplog)


μ() = ContinuousNetwork(Chain(normalize(s_mean, s_std), Dense(S.dims[1], 64, tanh), Dense(64, 32, tanh), Dense(32, adim)))
log_std() = -0.5f0*ones(Float32, adim)

𝒮_bc = BC(π=GaussianPolicy(μ(), log_std()), 
          𝒟_expert=expert_trajectories, 
          S=S, 
          # window=500,
          opt=(epochs=100000, early_stopping=(args...)->false,), 
          log=(period=1000,), 
          max_steps=1000)
solve(𝒮_bc, mdp)



A() = ContinuousNetwork(Chain(normalize(s_mean, s_std), Dense(S.dims[1], 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 6, init=Flux.orthogonal)))
D() = ContinuousNetwork(Chain(normalize(sa_mean, sa_std), Dense(idim, 256, relu, init=Flux.orthogonal), 
            Dense(256, 256, relu, init=Flux.orthogonal), 
            Dense(256, 1, init=Flux.orthogonal)))
            
𝒮_advil = AdVIL(π=A(), D=D(), 𝒟_expert=expert_trajectories, S=S, a_opt=(epochs=100000, optimizer=ADAM(8f-4), batch_size=1024), d_opt=(optimizer=ADAM(8e-4),), max_steps=1000, log=(period=1,))
solve(𝒮_advil, mdp)

