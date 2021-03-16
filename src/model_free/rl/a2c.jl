# A2C loss
a2c_loss(;λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0) = (π, 𝒟; info = Dict()) -> a2c_loss(π, 𝒟[:s], 𝒟[:a], 𝒟[:advantage], 𝒟[:logprob], λₚ, λₑ, info)

function a2c_loss(π, s, a, A, old_probs, λₚ, λₑ, info = Dict())
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

# Build an A2C solver
A2C(;π::ActorCritic, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), λₚ::Float32 = 1f0, λₑ::Float32 = 0.1f0, kwargs...) = 
    OnPolicySolver(;
        π = π,
        log = LoggerParams(;dir = "log/a2c", log...),
        a_opt = TrainingParams(;loss = a2c_loss(λₚ=λₚ, λₑ=λₑ), early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
        c_opt = TrainingParams(;loss = (π, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
        post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
        kwargs...)
    



