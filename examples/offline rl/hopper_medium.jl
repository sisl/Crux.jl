using Crux, Flux, POMDPs, POMDPGym, Distributions, POMDPPolicies, HDF5
init_mujoco_render() # Required for visualization

include("d4rl_utils.jl")

# Construct the Mujoco environment
mdp = GymPOMDP(Symbol("hopper-medium"), version=:v0)
S = state_space(mdp)
adim = length(POMDPs.actions(mdp)[1])
amin = -1*ones(Float32, adim)
amax = 1*ones(Float32, adim)

filename = "/home/anthonycorso/.d4rl/datasets/hopper_medium.hdf5"

## (Load from bson) get expert trajectories
𝒟_train = h5toEB(filename) 
# μ_s = mean(𝒟_train[:s], dims=2)[:]
# σ_s = std(𝒟_train[:s], dims=2)[:] .+ 1f-3
# S = state_space(mdp, μ=μ_s, σ=σ_s)
S = state_space(mdp)

## Behavioral Cloning 
A() = ContinuousNetwork(Chain(Dense(S.dims[1], 256, relu), Dense(256, 256, relu), Dense(256, 6, tanh)))
function SAC_A()
    base = Chain(Dense(S.dims[1], 256, relu), Dense(256, 256, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(256, adim)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(256, adim)))
    SquashedGaussianPolicy(mu, logΣ)
end

Q() = ContinuousNetwork(Chain(Dense(S.dims[1] + adim, 256, relu), Dense(256, 256, relu), Dense(256, 1))) 


## Behavioral Cloning
𝒮_bc = BC(π=SAC_A(), 
          𝒟_demo=𝒟_train, 
          S=S, 
          opt=(epochs=100000, batch_size=1024), 
          log=(period=100, fns=[log_undiscounted_return(1)]),
          max_steps=1000)
solve(𝒮_bc, mdp)

## Batch SAC 
𝒮_sac = BatchSAC(;π=ActorCritic(SAC_A(), DoubleNetwork(Q(), Q())),
                     𝒟_train=𝒟_train, 
                     S=S, 
                     log = (;period=100, fns=[log_undiscounted_return(1)]))
solve(𝒮_sac, mdp)

## CQL
𝒮_cql = CQL(;π=ActorCritic(SAC_A(), DoubleNetwork(Q(), Q())), 
             𝒟_train=𝒟_train, 
             S=S, 
             max_steps=1000,
             CQL_α_thresh = 10f0,
             a_opt=(epochs=100, batch_size=256, optimizer=Adam(1e-4)),
             c_opt=(batch_size=256, optimizer=Adam(3f-4),),
             SAC_α_opt=(batch_size=256, optimizer=Adam(1f-4),),
             CQL_α_opt=(batch_size=256, optimizer=Adam(1f-4),),
             log=(period=500, fns=[log_undiscounted_return(1)]),)
solve(𝒮_cql, mdp)



