using POMDPs, Crux, Flux, POMDPGym, BSON
import POMDPPolicies:FunctionPolicy
import Distributions:Uniform
using Random
using Distributions

## Pendulum
mdp = PendulumMDP()
as = [actions(mdp)...]
amin = [-1f0]
amax = [1f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
S = state_space(mdp, σ=[6.3f0, 8f0])

# get expert trajectories
expert_trajectories = BSON.load("/home/anthonycorso/.julia/dev/Crux/examples/il/expert_data/pendulum.bson")[:data]
expert_perf = sum(expert_trajectories[:r]) / length(episodes(expert_trajectories))
expert_trajectories[:r] .=1

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
QSA_SN() = ContinuousNetwork(Chain(DenseSN(3, 64, relu), DenseSN(64, 64, relu), DenseSN(64, 1)))
V() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(Dense(2, 64, relu, init=Flux.orthogonal), Dense(64, 64, relu, init=Flux.orthogonal), Dense(64, 1, tanh, init=Flux.orthogonal), x -> 2f0 * x), 1)
G() = GaussianPolicy(A(), zeros(Float32, 1))

function SAC_A()
    base = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(64, 1)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(64, 1)))
    SquashedGaussianPolicy(mu, logΣ)
end

# This currently doesn't work for some reason
𝒮_gail = GAIL(D=QSA_SN(), gan_loss=GAN_BCELoss(), 𝒟_demo=expert_trajectories, solver=PPO, π=ActorCritic(G(), V()), S=S, N=1000000, ΔN=1024)
solve(𝒮_gail, mdp)

𝒮_bc = BC(π=A(), 𝒟_demo=expert_trajectories, S=S, opt=(epochs=100,), log=(period=10,))
solve(𝒮_bc, mdp)

𝒮_advil = AdVIL(π=ActorCritic(A(),QSA()), 𝒟_demo=expert_trajectories, S=S, a_opt=(epochs=1000, optimizer=ADAM(8f-4), batch_size=1024), c_opt=(optimizer=ADAM(8e-4),), max_steps=100, log=(period=10,))
solve(𝒮_advil, mdp)


𝒮_sqil = SQIL(π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), 
              S=S,
              𝒟_demo=expert_trajectories,
              max_steps=100,
              N=30000,
              buffer_size=Int(1e4),
              c_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=128, optimizer=ADAM(1e-3)),
              π_explore=GaussianNoiseExplorationPolicy(0.2f0, a_min=[-2.0], a_max=[2.0]))

solve(𝒮_sqil, mdp)

Crux.set_crux_warnings(false)
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


𝒮_ASAF = ASAF(π=G(), 
              S=S, 
              ΔN=2000, 
              𝒟_demo=expert_trajectories,
              N=50000,
              max_steps=100,
              a_opt=(batch_size=256, optimizer=Flux.Optimise.Optimiser(Flux.ClipValue(1f0), ADAM(1e-3)), epochs=10))

solve(𝒮_ASAF, mdp)

using Plots
p = plot_learning([𝒮_gail, 𝒮_bc, 𝒮_advil, 𝒮_sqil, 𝒮_ASAF], title="Pendulum Swingup Imitation Learning Curves", labels=["GAIL", "BC", "AdVIL", "SQIL", "AdRIL", "ASAF"], legend=:right)
plot!(p, [1,100000], [expert_perf, expert_perf], color=:black, label="expert")

savefig("pendulum_benchmark.pdf")

