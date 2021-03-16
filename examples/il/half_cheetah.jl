using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies, BSON
init_mujoco_render() # Required for visualization

# Construct the Mujoco environment
mdp = GymPOMDP(:HalfCheetah, version = :v3)
mdplog = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

# get expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/half_cheetah.bson")[:data]

sum(expert_trajectories.data[:r]) / 10
expert_trajectories[:t]
expert_trajectories[:expert_val]

# Initializations that match the default PyTorch initializations
Winit(out, in) = Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out, in))
binit(in) = (out) -> Float32.(rand(Uniform(-sqrt(1/in), sqrt(1/in)), out))

# Networks for on-policy algorithms
idim = S.dims[1] + adim
# μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), 
#                               Dense(64, 32, tanh, initW = Winit, initb = binit(64)), 
#                               Dense(32, adim, initW = Winit, initb = binit(32))))
# D() = ContinuousNetwork(Chain(DenseSN(idim, 64, relu), DenseSN(64, 64, relu), DenseSN(64, 1)))
# V() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), 
#                               Dense(64, 32, initW = Winit, initb = binit(64)), 
#                              Dense(32, 1, initW = Winit, initb = binit(32))))
# log_std() = -0.5f0*ones(Float32, adim)
# 
# # Networks for off-policy algorithms
# Q() = ContinuousNetwork(Chain(Dense(idim, 256, relu, initW = Winit, initb = binit(idim)), 
#             Dense(256, 256, relu, initW = Winit, initb = binit(256)), 
#             Dense(256, 1, initW = Winit, initb = binit(256))) )
# 
# 
# 
# A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, initW = Winit, initb = binit(idim)), 
#             Dense(256, 256, relu, initW = Winit, initb = binit(256)), 
#             Dense(256, 6, initW = Winit, initb = binit(256))))
# A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu, initW = Winit, initb = binit(idim)), 
#             Dense(256, 256, relu, initW = Winit, initb = binit(256)), 
#             Dense(256, 6, initW = Winit, initb = binit(256))))


## Setup params
shared = (max_steps=1000, N=Int(1e6), S=S)

# Solve with PPO-GAIL
𝒮_gail = GAIL(;D=D(), 
              gan_loss=GAN_LSLoss(), 
              𝒟_expert=expert_trajectories, 
              solver=PPO, 
              π=ActorCritic(GaussianPolicy(μ(), log_std()), V()), 
              ΔN=4000, 
              a_opt=(batch_size=4000, epochs=80, optimizer=ADAM(3e-4)),
              shared...)
solve(𝒮_gail, mdp)

# solve with valueDICE
D() = ContinuousNetwork(Chain(Dense(idim, 256, relu, initW=Flux.orthogonal, initb = binit(idim)), 
            Dense(256, 256, relu, initW=Flux.orthogonal, initb = binit(256)), 
            Dense(256, 1, initW=Flux.orthogonal, bias=false)))
            
function SAC_A()
    base = Chain(Dense(S.dims[1], 256, relu, initW=Flux.orthogonal, initb = binit(idim)), 
                Dense(256, 256, relu, initW=Flux.orthogonal, initb = binit(256)))
    mu = ContinuousNetwork(Chain(base..., Dense(256, 6, initW=Flux.orthogonal, initb = binit(256))))
    logΣ = ContinuousNetwork(Chain(base..., Dense(256, 6, initW=Flux.orthogonal, initb = binit(256))))
    SquashedGaussianPolicy(mu, logΣ)
end

𝒮_valueDICE = ValueDICE(;π=ActorCritic(SAC_A(), D()),
                        𝒟_expert=expert_trajectories, 
                        max_steps=1000, 
                        N=Int(1e6), 
                        S=S,
                        α=0.1,
                        buffer_size=Int(1e6), 
                        buffer_init=200, 
                        c_opt=(batch_size=5, optimizer=ADAM(1e-3)), 
                        a_opt=(batch_size=5, optimizer=ADAM(1e-5)))
                        
solve(𝒮_valueDICE, mdp, mdplog)


μ() = ContinuousNetwork(Chain(Dense(S.dims[1], 64, tanh, initW = Winit, initb = binit(S.dims[1])), 
                              Dense(64, 32, tanh, initW = Winit, initb = binit(64)), 
                              Dense(32, adim, initW = Winit, initb = binit(32))))
log_std() = -0.5f0*ones(Float32, adim)

𝒮_bc = BC(π=GaussianPolicy(μ(), log_std()), 
          𝒟_expert=expert_trajectories, 
          S=S, 
          opt=(epochs=60000,), 
          log=(period=10,), 
          max_steps=1000)
solve(𝒮_bc, mdp)


s_mean = dropdims(mean(expert_trajectories[:s], dims=2), dims=2)
s_std = dropdims(std(expert_trajectories[:s], dims=2), dims=2)

sa_mean = vcat(s_mean, zeros(Float32, 6))
sa_std = vcat(s_std, ones(Float32, 6))

function normalize(μ, σ²)
    (x) -> (x .- μ) ./ σ²
end    

A() = ContinuousNetwork(Chain(normalize(s_mean, s_std), Dense(S.dims[1], 256, relu, initW=Flux.orthogonal, initb = binit(idim)), 
            Dense(256, 256, relu, initW=Flux.orthogonal, initb = binit(256)), 
            Dense(256, 6, initW=Flux.orthogonal, initb = binit(256))))
D() = ContinuousNetwork(Chain(normalize(sa_mean, sa_std), Dense(idim, 256, relu, initW=Flux.orthogonal, initb = binit(idim)), 
            Dense(256, 256, relu, initW=Flux.orthogonal, initb = binit(256)), 
            Dense(256, 1, initW=Flux.orthogonal, initb = binit(256))))
            
𝒮_advil = AdVIL(π=A(), D=D(), 𝒟_expert=expert_trajectories, S=S, a_opt=(epochs=100000, optimizer=ADAM(8f-6), batch_size=1024), d_opt=(optimizer=ADAM(8e-4),), max_steps=1000, log=(period=100,))
solve(𝒮_advil, mdp)

