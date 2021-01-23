@with_kw mutable struct DDPGSolver <: Solver
    π::ActorCritic{ContinuousNetwork, ContinuousNetwork}
    S::AbstractSpace # state space
    A::AbstractSpace = action_space(π) # action space
    N::Int = 1000 # number of training iterations
    rng::AbstractRNG = Random.GLOBAL_RNG
    exploration_policy::ExplorationPolicy = GaussianNoiseExplorationPolicy(0.1f0, rng = rng)
    critic_loss::Function = Flux.Losses.mse # critic loss function
    regularizer_actor = (θ) -> 0
    regularizer_critic = (θ) -> 0
    opt_actor = Flux.Optimiser(ClipNorm(1f0), ADAM(1f-4)) # optimizer for the actor
    opt_critic =Flux.Optimiser(ClipNorm(1f0), ADAM(1f-3)) # optimizer for the critic
    τ::Float32 = 0.005f0 # polyak averaging parameters used when updating target networks
    batch_size::Int = 100
    max_steps::Int = 100
    eval_eps::Int = 10
    ΔN::Int = 4
    epochs::Int = ΔN
    buffer_size = 1000
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size)
    buffer_init::Int = max(batch_size, 200)
    log::Union{Nothing, LoggerParams} = LoggerParams(dir="log/ddpg", period=500)
    π⁻::ActorCritic{ContinuousNetwork, ContinuousNetwork} = deepcopy(π) # Target network
    device = device(π)
    i::Int = 0
end

DDPG_target(π, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(π, 𝒟[:sp], action(π, 𝒟[:sp]))

# T. P. Lillicrap, et al., "Continuous control with deep reinforcement learning", ICLR 2016.
function POMDPs.solve(𝒮::DDPGSolver, mdp)
    # Initialize replay buffer 𝒟
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.batch_size, device=𝒮.device)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps, exploration_policy=𝒮.exploration_policy)

    # Logging: log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps=𝒮.eval_eps))

    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
    
    # for t = 1, T do
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Select action aₜ = μ(sₜ | θᵘ) + 𝒩ₜ according to the current policy and exploration noise
        # Execute action aₜ and observe reward rₜ and observe new state sₜ₊₁
        # Store transition (sₜ, aₜ, rₜ, sₜ₊₁) in 𝒟
        push!(𝒮.buffer, steps!(s, explore=true, i=𝒮.i, Nsteps=𝒮.ΔN))

        infos = []
        for tᵢ in 1:𝒮.epochs
            # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
            rand!(𝒮.rng, 𝒟, 𝒮.buffer, i=𝒮.i)

            # Set yᵢ = rᵢ + γQ′(sᵢ₊₁, μ′(sᵢ₊₁ | θᵘ′) | θᶜ′)
            y = DDPG_target(𝒮.π⁻, 𝒟, γ)
            

            # Update critic by minimizing the loss: ℒ = 1/𝑁 Σᵢ (yᵢ - Q(sᵢ, aᵢ, | θᶜ))²
            info_c = train!(𝒮.π.C, (;kwargs...) -> td_loss(𝒮.π, 𝒟, y, 𝒮.critic_loss; kwargs...), 𝒮.opt_critic, 
                            loss_sym = :critic_loss, 
                            grad_sym = :critic_grad_norm, 
                            regularizer = 𝒮.regularizer_critic)
                            
            # Update the actor policy using the sampled policy gradient (using gradient ascent, note minus sign):
            # ∇_θᵘ 𝐽 ≈ 1/𝑁 Σᵢ ∇ₐQ(s, a | θᶜ)|ₛ₌ₛᵢ, ₐ₌ᵤ₍ₛᵢ₎ ∇_θᵘ μ(s | θᵘ)|ₛᵢ
            info_a = train!(𝒮.π.A, (;kwargs...) -> -mean(value(𝒮.π, 𝒟[:s], action(𝒮.π, 𝒟[:s]))), 𝒮.opt_actor, 
                            loss_sym = :actor_loss, 
                            grad_sym = :actor_grad_norm,
                            regularizer = 𝒮.regularizer_actor)
                            
            # Merge the loss information and store it
            push!(infos, merge(info_a, info_c))

            # Update the target networks:
            # θᶜ′ ⟵ τθᶜ + (1 - τ)θᶜ′
            # θᵘ′ ⟵ τθᵘ + (1 - τ)θᵘ′
            polyak_average!(𝒮.π⁻, 𝒮.π, 𝒮.τ)
        end
        

        # Logging: Log results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps=𝒮.eval_eps),
                                        aggregate_info(infos),
                                        log_exploration(𝒮.exploration_policy, 𝒮.i))
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

