@with_kw mutable struct PGSolver <: Solver
    π::Policy
    S::AbstractSpace
    A::AbstractSpace = action_space(π)
    loss
    N::Int64 = 1000
    ΔN::Int = 200
    λ_gae::Float32 = 0.95
    batch_size::Int = 64
    batch_size_critic::Int = batch_size
    epochs::Int = 100
    epochs_critic::Int = epochs
    max_steps::Int64 = 100
    eval_eps::Int = 10
    opt = ADAM(3e-4)
    opt_critic = deepcopy(opt)
    loss_critic = (π, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return])
    regularizer = (π) -> 0
    regularizer_critic = regularizer
    early_stopping = (info) -> (info[:kl] > 0.015)
    required_columns = π isa ActorCritic ? [:return, :advantage, :logprob] : [:return, :logprob]
    normalize_advantage = (:advantage in required_columns) ? true : false
    rng::AbstractRNG = Random.GLOBAL_RNG
    log = LoggerParams(dir = "log/actor_critic", period = 500)
    device = device(π)
    i::Int64 = 0
end

# REINFORCE loss
reinforce() = (π, 𝒟; info = Dict()) -> reinforce(π, 𝒟[:s], 𝒟[:a], 𝒟[:return], 𝒟[:logprob], info)
function reinforce(π, s, a, G, old_probs, info = Dict())
    new_probs = logpdf(π, s, a)
    
    ignore() do
        info[:entropy] = mean(entropy(π, s))
        info[:kl] = mean(old_probs .- new_probs)
    end 
    
    -mean(new_probs .* G)
end

# A2C loss
a2c(;λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟; info = Dict()) -> a2c(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:logprob], λₚ, λₑ, info)

function a2c(π, s, a, A, old_probs, λₚ, λₑ, info = Dict())
    new_probs = logpdf(π, s, a)
    p_loss = -mean(new_probs .* A)
    e_loss = -mean(entropy(π, s))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(old_probs .- new_probs)
    end 
    
    λₚ*p_loss + λₑ*e_loss
end

# PPO loss
ppo(;ϵ::Float32 = 0.2f0, λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟; info = Dict()) -> ppo(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:logprob], ϵ, λₚ, λₑ, info)

function ppo(π, s, a, A, old_probs, ϵ, λₚ, λₑ, info = Dict())
    new_probs = logpdf(π, s, a) 
    r = exp.(new_probs .- old_probs)

    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - ϵ), (1f0 + ϵ)) .* A))
    e_loss = -mean(entropy(π, s))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(old_probs .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + ϵ) .| (r .< 1 - ϵ)) / length(r)
    end 
    λₚ*p_loss + λₑ*e_loss
end

function POMDPs.solve(𝒮::PGSolver, mdp)
    # Construct the experience buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device=𝒮.device)
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.π, 𝒮.S, required_columns=𝒮.required_columns, λ=λ, max_steps=𝒮.max_steps, rng=𝒮.rng, exploration_policy=𝒮.π)

    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps=𝒮.eval_eps))

    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions
        push!(𝒟, steps!(s, Nsteps = 𝒮.ΔN, reset = true, explore = true))
        
        # Normalize the advantage
        𝒮.normalize_advantage && (𝒟[:advantage] .= whiten(𝒟[:advantage]))
        
        # Train the policy (using batches)
        info = train!(𝒮.π, 𝒮.loss, 𝒮.batch_size, 𝒮.opt, 𝒟, 
                        epochs = 𝒮.epochs, 
                        rng = 𝒮.rng, 
                        regularizer = 𝒮.regularizer, 
                        early_stopping = 𝒮.early_stopping,
                        loss_sym = :policy_loss,
                        grad_sym = :policy_grad_norm)
        
        # Train the value function (if actor critic)
        if 𝒮.π isa ActorCritic
            info_c = train!(𝒮.π, 𝒮.loss_critic, 𝒮.batch_size_critic, 𝒮.opt, 𝒟, 
                            epochs = 𝒮.epochs_critic,
                            rng = 𝒮.rng, 
                            regularizer = 𝒮.regularizer_critic,
                            loss_sym = :value_loss, 
                            grad_sym = :value_grad_norm)
            merge!(info, info_c)
        end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps = 𝒮.eval_eps), info)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end
