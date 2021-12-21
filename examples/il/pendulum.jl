using POMDPs, Crux, Flux, POMDPGym, BSON
import POMDPPolicies:FunctionPolicy
import Distributions:Uniform
using Random
using Distributions
using Plots

## Pendulum
mdp = PendulumPOMDP()
as = [actions(mdp)...]
amin = [-2f0]
amax = [2f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
S = state_space(mdp, σ=[3.14f0, 8f0])

# get expert trajectories
expert_trajectories = BSON.load("/Users/anthonycorso/.julia/dev/Crux/examples/il/expert_data/pendulum.bson")[:data]
expert_perf = sum(expert_trajectories[:r]) / length(episodes(expert_trajectories))
expert_trajectories[:r] .= 1

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
QSA_SN(output=1) = ContinuousNetwork(Chain(DenseSN(3, 64, relu), DenseSN(64, 64, relu), DenseSN(64, 2), Dense(2,output)))
V() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)
SG() = SquashedGaussianPolicy(ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))), zeros(Float32, 1), 2f0)
G() = GaussianPolicy(A(), zeros(Float32, 1))

D_SN(output=1) = ContinuousNetwork(Chain(DenseSN(3, 100, tanh), DenseSN(100,100, tanh), DenseSN(100,output)))

## On-Policy GAIL - This currently doesn't work for some reason
𝒮_gail_on = OnPolicyGAIL(D=QSA(),
                         γ=discount(mdp),
                         gan_loss=GAN_BCELoss(), 
                         𝒟_demo=expert_trajectories, 
                         solver=PPO, 
                         π=ActorCritic(G(), V()), 
                         S=S, 
                         N=1000000,
                         d_opt=(batch_size=1024, epochs=80),
                         ΔN=1024)
solve(𝒮_gail_on, mdp)

## Off-Policy GAIL
𝒮_gail = OffPolicyGAIL(D=D_SN(2), 
                       𝒟_demo=expert_trajectories, 
                       solver=TD3, 
                       π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), 
                       S=S,
                       ΔN=50,
                       N=30000,
                       buffer_size=Int(1e4),
                       c_opt=(batch_size=512, optimizer=ADAM(1e-3)),
                       a_opt=(optimizer=ADAM(1e-3),),
                       d_opt=(epochs=1, batch_size=256),
                       π_explore=GaussianNoiseExplorationPolicy(0.2f0, a_min=[-2.0], a_max=[2.0])
                       )
solve(𝒮_gail, mdp)


## Behavioral Cloning 
𝒮_bc = BC(π=G(), 𝒟_demo=expert_trajectories, S=S, opt=(epochs=100,), log=(period=100,))
solve(𝒮_bc, mdp)

## Advil
𝒮_advil = AdVIL(π=ActorCritic(A(),QSA()), 𝒟_demo=expert_trajectories, S=S, a_opt=(epochs=1000, optimizer=ADAM(8f-4), batch_size=1024), c_opt=(optimizer=ADAM(8e-4),), max_steps=100, log=(period=100,))
solve(𝒮_advil, mdp)

## SQIL
𝒮_sqil = SQIL(π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), 
              S=S,
              𝒟_demo=expert_trajectories,
              max_steps=100,
              N=30000,
              buffer_size=Int(1e4),
              c_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              solver=TD3,
              π_explore=GaussianNoiseExplorationPolicy(0.2f0, a_min=[-2.0], a_max=[2.0]))
solve(𝒮_sqil, mdp)

## Adril
𝒮_adril = AdRIL(π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), 
              S=S,
              𝒟_demo=expert_trajectories,
              max_steps=100,
              N=30000,
              buffer_size=Int(1e4),
              c_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              π_explore=GaussianNoiseExplorationPolicy(0.2f0, a_min=[-2.0], a_max=[2.0]))
solve(𝒮_adril, mdp)


## ASAF
𝒮_ASAF = ASAF(π=SG(), 
              S=S, 
              ΔN=2000, 
              𝒟_demo=expert_trajectories,
              N=50000,
              max_steps=100,
              a_opt=(batch_size=256, optimizer=Flux.Optimise.Optimiser(Flux.ClipValue(1f0), ADAM(1e-3)), epochs=10))
solve(𝒮_ASAF, mdp)


p = plot_learning([𝒮_gail_on, 𝒮_gail, 𝒮_bc, 𝒮_advil, 𝒮_sqil, 𝒮_adril,𝒮_ASAF], title="Pendulum Swingup Imitation Learning Curves", labels=["On Policy GAIL", "Off-Policy GAIL", "BC", "AdVIL", "SQIL", "AdRIL", "ASAF"], legend=:right)
plot!(p, [1,50000], [expert_perf, expert_perf], color=:black, label="expert")

savefig("examples/il/pendulum_benchmark.pdf")

