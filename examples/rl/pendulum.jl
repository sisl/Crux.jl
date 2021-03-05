using POMDPs, Crux, Flux, POMDPGym
import POMDPPolicies:FunctionPolicy
import Distributions:Uniform
using Random
using Distributions

## Pendulum
mdp = PendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.])
as = [actions(mdp)...]
amin = [-1f0]
amax = [1f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
S = state_space(mdp)

# Define the networks we will use
QSA() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
QS() = DiscreteNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1)))
A() = ContinuousNetwork(Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1, tanh), x -> 2f0 * x), 1)

G() = GaussianPolicy(A(), zeros(Float32, 1))
function SAC_A()
    base = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(64, 1)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(64, 1)))
    SquashedGaussianPolicy(mu, logΣ)
end


# Solve with REINFORCE (Generally doesn't learn much, ~15 secs)
𝒮_reinforce = REINFORCE(π=G(), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
@time π_reinforce = solve(𝒮_reinforce, mdp)

# Solve with A2C (Generally doesn't learn much, ~1 min)
𝒮_a2c = A2C(π=ActorCritic(G(), V()), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
@time π_a2c = solve(𝒮_a2c, mdp)

# Solve with PPO (gets to > -200 reward, ~1.5 min)
𝒮_ppo = PPO(π=ActorCritic(G(), V()), S=S, N=100000, ΔN=2048, a_opt=(batch_size=512,))
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
              
# Solver with DDPG (gets to > -200 reward, ~1 min)
𝒮_ddpg = DDPG(;π=ActorCritic(A(), QSA()), off_policy...)
@time π_ddpg = solve(𝒮_ddpg, mdp)

# Solve with TD3 (didn't learn much, ~1.5 min)
𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_td3 = solve(𝒮_td3, mdp)

# Solve with TD3 (didn't learn much, ~1.5 min)
𝒮_sac = SAC(;π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)

# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_ddpg, 𝒮_td3, 𝒮_sac], title="Pendulum Swingup Training Curves", labels=["REINFORCE", "A2C", "PPO", "DQN", "DDPG", "TD3", "SAC"], legend=:right)

# Produce a gif with the final policy
gif(mdp, π_dqn, "pendulum.gif", max_steps=200)

