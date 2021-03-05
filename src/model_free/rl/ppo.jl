# PPO loss
ppo_loss(;ϵ::Float32 = 0.2f0, λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟; info = Dict()) -> ppo_loss(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:logprob], ϵ, λₚ, λₑ, info)

function ppo_loss(π, s, a, A, old_probs, ϵ, λₚ, λₑ, info = Dict())
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
# Build an A2C solver
PPO(;π::ActorCritic, ϵ::Float32 = 0.2f0, λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) = 
    OnPolicySolver(;
        π = π,
        log = LoggerParams(;dir = "log/ppo", log...),
        a_opt = TrainingParams(;loss = ppo_loss(ϵ=ϵ, λₚ=λₚ, λₑ=λₑ), early_stopping = (info) -> (info[:kl] > 0.015), name = "actor_", a_opt...),
        c_opt = TrainingParams(;loss = (π, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
        post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
        kwargs...)
    



