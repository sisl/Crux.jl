using POMDPs, Crux, Flux, POMDPGym
import POMDPPolicies:FunctionPolicy
import Distributions:Uniform
using Random
using Distributions

## Pendulum
mdp = PendulumPOMDP(actions=[-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
amin = [-2f0]
amax = [2f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
S = state_space(mdp, σ=[3.14f0, 8f0])

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
QS() = DiscreteNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)
SG() = SquashedGaussianPolicy(ContinuousNetwork(Chain(Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))), zeros(Float32, 1), 2f0)


# Solve with REINFORCE (Generally doesn't learn much, ~15 secs)
𝒮_reinforce = REINFORCE(π=SG(), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C (Generally doesn't learn much, ~1 min)
𝒮_a2c = A2C(π=ActorCritic(SG(), V()), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
@time π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO (gets to > -200 reward, ~1.5 min)
𝒮_ppo = PPO(π=ActorCritic(SG(), V()), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,), λe=0f0)
@time π_ppo = solve(𝒮_ppo, mdp)

# Solve with DQN (gets to > -200 reward, ~30 sec)
𝒮_dqn = DQN(π=QS(), S=S, N=30000)
@time π_dqn = solve(𝒮_dqn, mdp)

off_policy = (S=S,
              ΔN=50,
              N=30000,
              buffer_size=Int(5e5),
              buffer_init=1000,
              c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              π_explore=FirstExplorePolicy(1000, rand_policy, GaussianNoiseExplorationPolicy(0.5f0, a_min=[-2.0], a_max=[2.0])))
              
# Solver with DDPG
𝒮_ddpg = DDPG(;π=ActorCritic(A(), QSA()), off_policy...)
@time π_ddpg = solve(𝒮_ddpg, mdp)

# Solve with TD3
𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_td3 = solve(𝒮_td3, mdp)

# Solve with SAC
𝒮_sac = SAC(;π=ActorCritic(SG(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)


# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_ddpg, 𝒮_td3, 𝒮_sac], title="Pendulum Swingup Training Curves", labels=["REINFORCE", "A2C", "PPO", "DQN", "DDPG", "TD3", "SAC"], legend=:right)
Crux.savefig("examples/rl/pendulum_benchmark.pdf")

# Produce a gif with the final policy
gif(mdp, π_dqn, "pendulum.gif", max_steps=200)

## Save data for imitation learning
# using BSON
# s = Sampler(mdp, 𝒮_ppo.π, max_steps=200, required_columns=[:t])
# 
# data = steps!(s, Nsteps=10000)
# sum(data[:r])/50
# data[:expert_val] = ones(Float32, 1, 10000)
# 
# data = ExperienceBuffer(data)
# BSON.@save "examples/il/expert_data/pendulum.bson" data

