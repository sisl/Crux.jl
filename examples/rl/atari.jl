using POMDPs, POMDPPolicies, POMDPGym, Crux
using Flux, Random

mdp = AtariPOMDP(:Pong, version = :v0)
S = ContinuousSpace((80,80,4), Float32) # Needs to be Float32 or the conv layers through an error on gpu
as = actions(mdp)

Q() = DiscreteNetwork(Chain(x->x ./ 255f0, Conv((8,8), 4=>16, relu, stride = 4), Conv((4,4), 16=>32, relu, stride = 2), flatten, Dense(2048, 256, relu), Dense(256, length(as))), as) |> gpu

𝒮 = DQN(π=Q(), S=S, N=5000000, buffer_size=10000, max_steps=1000, buffer_init=5000, log=(fns=[log_undiscounted_return(1)],), c_opt=(optimizer=Flux.Optimiser(ClipValue(1f0), Adam(1f-3)),))
solve(𝒮, mdp)

# Plot the learning curve
p = plot_learning(𝒮, title = "Pong Training Curve")

# Produce a gif with the final policy
gif(mdp, 𝒮.π, "pong.gif")

