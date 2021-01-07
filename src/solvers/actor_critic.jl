@with_kw mutable struct PGSolver <: Solver 
    π::Policy
    S::AbstractSpace
    A::AbstractSpace = action_space(π)
    loss
    N::Int64 = 1000
    ΔN::Int = 200
    λ_gae::Float32 = 0.95
    batch_size::Int = 64
    epochs::Int = 10
    max_steps::Int64 = 100
    eval_eps::Int = 10
    opt = ADAM(3e-4)
    required_columns = π isa ActorCritic ? [:return, :advantage, :logprob] : [:return]
    rng::AbstractRNG = Random.GLOBAL_RNG
    log = LoggerParams(dir = "log/actor_critic", period = 500)
    device = device(π)
    i::Int64 = 0
end

# REINFORCE loss
reinforce() = (π, 𝒟) -> reinforce(π, 𝒟[:s], 𝒟[:a], 𝒟[:return])
reinforce(π, s, a, G) = -mean(logpdf(π, s, a) .* G)

# A2C Loss
a2c(;λₚ::Float32 = 1f0, λᵥ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟) -> a2c(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:return], λₚ, λᵥ, λₑ)

function a2c(π, s, a, A, G, λₚ, λᵥ, λₑ)
        p_loss = -mean(logpdf(π, s, a) .* A)
        v_loss = mean((value(π, s) .- G).^2)
        e_loss = -mean(entropy(π, s))
        
        λₚ*p_loss + λᵥ*v_loss + λₑ*e_loss
end

# PPO Loss
ppo(;ϵ::Float32 = 0.2f0, λₚ::Float32 = 1f0, λᵥ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟) -> ppo(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:return], 𝒟[:logprob], ϵ, λₚ, λᵥ, λₑ)

function ppo(π, s, a, A, G, old_probs, ϵ, λₚ, λᵥ, λₑ)
        r = exp.(logpdf(π, s, a) .- old_probs)

        p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - ϵ), (1f0 + ϵ)) .* A))
        v_loss = mean((value(π, s) .- G).^2)
        e_loss = -mean(entropy(π, s))

        λₚ*p_loss + λᵥ*v_loss + λₑ*e_loss
end

function POMDPs.solve(𝒮::PGSolver, mdp)
    # Construct the experience buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device = 𝒮.device)
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, required_columns = 𝒮.required_columns, λ = 𝒮.λ_gae, max_steps = 𝒮.max_steps, rng = 𝒮.rng)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.ΔN, step = 𝒮.ΔN)
        # Sample transitions
        push!(𝒟, steps!(s, Nsteps = 𝒮.ΔN, reset = true))
        
        # Train the policy (using batches)
        losses, grads = train!(𝒮.π, (D) -> 𝒮.loss(𝒮.π, D), 𝒮.batch_size, 𝒮.opt, 𝒟, epochs = 𝒮.epochs, rng = 𝒮.rng)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                        log_loss(losses),
                                        log_gradient(grads))
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

