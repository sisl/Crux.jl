@with_kw mutable struct DDPGSolver <: Solver
    π::Policy # behavior policy
    π′::Policy # target policy
    S::AbstractSpace # state space
    A::AbstractSpace = action_space(π) # action space
    N::Int = 1000 # number of training iterations
    rng::AbstractRNG = Random.GLOBAL_RNG
    exploration_policy::ExplorationPolicy = GaussianNoiseExplorationPolicy(0.1)
    critic_loss::Function = Flux.Losses.mse # critic loss function
    regularizer = (θ) -> 0
    opt_actor = Flux.Optimiser(ClipValue(1f0), ADAM(1f-3)) # optimizer for the actor
    opt_critic = Flux.Optimiser(ClipValue(1f0), ADAM(1f-3)) # optimizer for the critic
    τ = 0.001 # polyak averaging parameters used when updating target networks
    batch_size::Int = 100
    epochs::Int = 1
    max_steps::Int = 100
    eval_eps::Int = 10
    Δtrain::Int = 50
    buffer_size = 1000
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size)
    buffer_init::Int = max(batch_size, 200)
    log::Union{Nothing, LoggerParams} = LoggerParams(dir="log/ddpg", period=500)
    device = device(π)
    i::Int = 0
end

# T. P. Lillicrap, et al., "Continuous control with deep reinforcement learning", ICLR 2016.
function POMDPs.solve(𝒮::DDPGSolver, mdp)
    # Initialize replay buffer 𝒟
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.batch_size, device=𝒮.device)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π.A, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps, exploration_policy=𝒮.exploration_policy)

    # Logging: log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps=𝒮.eval_eps))

    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i)

    # for t = 1, T do
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.Δtrain, step=𝒮.Δtrain)
        # Select action aₜ = μ(sₜ | θᵘ) + 𝒩ₜ according to the current policy and exploration noise
        # Execute action aₜ and observe reward rₜ and observe new state sₜ₊₁
        # Store transition (sₜ, aₜ, rₜ, sₜ₊₁) in 𝒟
        push!(𝒮.buffer, steps!(s, explore=true, i=𝒮.i, Nsteps=𝒮.Δtrain))

        local actor_losses
        local actor_grads

        for tᵢ in 1:𝒮.Δtrain
            # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
            rand!(𝒮.rng, 𝒟, 𝒮.buffer, i=𝒮.i)

            # Set yᵢ = rᵢ + γQ′(sᵢ₊₁, μ′(sᵢ₊₁ | θᵘ′) | θᶜ′)
            y = target(𝒮.π′.A, 𝒮.π′.C, 𝒟, γ)

            # Update critic by minimizing the loss: ℒ = 1/𝑁 Σᵢ (yᵢ - Q(sᵢ, aᵢ, | θᶜ))²
            critic_losses, critic_grads = train!(𝒮.π.C, () -> 𝒮.critic_loss(value(𝒮.π.C, 𝒟[:s], 𝒟[:a]), y, agg=mean), 𝒮.opt_critic)

            # Update the actor policy using the sampled policy gradient (using gradient ascent, note minus sign):
            # ∇_θᵘ 𝐽 ≈ 1/𝑁 Σᵢ ∇ₐQ(s, a | θᶜ)|ₛ₌ₛᵢ, ₐ₌ᵤ₍ₛᵢ₎ ∇_θᵘ μ(s | θᵘ)|ₛᵢ
            actor_losses, actor_grads = train!(𝒮.π.A, () -> -mean(value(𝒮.π.C, 𝒟[:s], action(𝒮.π.A, 𝒟[:s]))), 𝒮.opt_actor)

            # Update the target networks:
            # θᶜ′ ⟵ τθᶜ + (1 - τ)θᶜ′
            # θᵘ′ ⟵ τθᵘ + (1 - τ)θᵘ′
            copyto!(𝒮.π′.C, 𝒮.π.C, 𝒮.τ)
            copyto!(𝒮.π′.A, 𝒮.π.A, 𝒮.τ)
        end

        # Logging: Log results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.Δtrain, log_undiscounted_return(s, Neps=𝒮.eval_eps),
                                           log_loss(actor_losses),
                                           log_gradient(actor_grads),
                                           log_exploration(𝒮.exploration_policy, 𝒮.i))
    end
    𝒮.i += 𝒮.Δtrain
    𝒮.π
end
