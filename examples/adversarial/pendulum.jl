using POMDPs, Crux, Flux, POMDPGym, Random, Distributions

## Pendulum
mdp = AdditiveAdversarialMDP(InvertedPendulumMDP(), Normal(0, 0.2))
amin = [-2f0]
amax = [2f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
S = state_space(mdp, σ=[6.3f0, 8f0])

