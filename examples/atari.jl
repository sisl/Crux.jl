using POMDPs, POMDPPolicies, POMDPGym, Crux, Plots, ImageCore
using Flux, Random

mdp = AtariPOMDP(:Pong, version = :v0)

s = rand(initialstate(mdp))
s, o, r = gen(mdp, s, 1)
as = actions(mdp)
s_dim = size(o)[1:3]
a_dim = length(as)

Q = Chain(x->x ./ 255f0, Conv((8,8), 4=>16, relu, stride = 4), Conv((4,4), 16=>32, relu, stride = 2), flatten, Dense(2048, 256, relu), Dense(256, length(as))) |> gpu
policy = DQNPolicy(Q = Q, actions = as)


buffer = ExperienceBuffer(s_dim, a_dim, 100000, S = UInt8, prioritized = true)
𝒮 = DQNSolver(π = policy, sdim = s_dim, N=5000000, buffer = buffer, eval_eps = 1, max_steps = 1000, Δtarget_update = 10000, buffer_init = 5000)
solve(𝒮, mdp)

# episode_gif(mdp, policy, "out.gif", render = (s) -> plot(torgb(s)))

# s = Sampler(mdp, policy, s_dim, a_dim, max_steps = 1000)
# undiscounted_return(s, Neps = 3)