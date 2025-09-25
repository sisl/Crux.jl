using POMDPs
using POMDPGym
import POMDPModels
using Test
using Crux
using Flux
using Random
using BSON

mdp = GymPOMDP(:CartPole)
as = actions(mdp)
S = state_space(mdp)

# Flux network: Map states to actions
A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 1)))

solver_reinforce = REINFORCE(S=S, π=A())
policy_reinforce = solve(solver_reinforce, mdp)

solver_a2c = A2C(S=S, π=ActorCritic(A(), V()))
policy_a2c = solve(solver_a2c, mdp)

solver_ppo = PPO(S=S, π=ActorCritic(A(), V()))
policy_ppo = solve(solver_ppo, mdp)

p = plot_learning([solver_reinforce, solver_a2c, solver_ppo],
                  title="CartPole Training Curves",
                  labels=["REINFORCE", "A2C", "PPO"])

Crux.savefig(p, "test.pdf")

rm("test.pdf")

