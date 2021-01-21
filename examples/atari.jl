using POMDPs, POMDPPolicies, POMDPGym, Crux
using Flux, Random

mdp = AtariPOMDP(:Pong, version = :v0)
S = state_space(mdp)
as = actions(mdp)

Q() = Chain(x->x ./ 255f0, Conv((8,8), 4=>16, relu, stride = 4), Conv((4,4), 16=>32, relu, stride = 2), flatten, Dense(2048, 256, relu), Dense(256, length(as))) |> gpu
policy = DQNPolicy(Q(), as)


buffer = ExperienceBuffer(S, action_space(policy), 100000, prioritized = true)
𝒮 = DQNSolver(π = policy, S = S, N=5000000, buffer = buffer, eval_eps = 1, max_steps = 1000, Δtarget_update = 10000, buffer_init = 5000, opt = Flux.Optimiser(ClipValue(1f0), ADAM(1f-3)))
solve(𝒮, mdp)

# Plot the learning curve
p = plot_learning(𝒮, title = "Pong Training Curve")

# Produce a gif with the final policy
gif(mdp, 𝒮.π, "pong.gif")

