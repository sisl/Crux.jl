using Crux, Flux, POMDPs, POMDPGym
init_mujoco_render()

mdp = GymPOMDP(:HalfCheetah, version = :v3)
S = state_space(mdp)
adim = length(actions(mdp)[1])

μ() = Chain(Dense(S.dims[1], 100, relu), Dense(100, 50, relu), Dense(50, 25, relu), Dense(25, adim))
V() = Chain(Dense(S.dims[1], 100, relu), Dense(100, 50, relu), Dense(50, 25, relu), Dense(25, 1))
log_std() = zeros(Float32, adim)


# Solve with ppo
𝒮_ppo = PGSolver(π = ActorCritic(GaussianPolicy(μ = μ(), logΣ = log_std()), V()), 
                 S = S, 
                 max_steps = 500, 
                 loss = ppo(λₑ = 0f0),
                 ΔN = 2048,
                 epochs = 10,
                 N = 1200000, 
                 opt = Flux.Optimiser(ClipNorm(1f0), ADAM(1e-4)))
solve(𝒮_ppo, mdp)

p = plot_learning(𝒮_ppo, title = "HalfCheetah Training Curves")

# Produce a gif with the final policy
gif(mdp, 𝒮_ppo.π, "mujoco.gif")

