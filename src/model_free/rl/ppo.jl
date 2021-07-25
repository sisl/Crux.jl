# PPO loss
function ppo_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a]) 
    r = exp.(new_probs .- 𝒟[:logprob])
    
    A = 𝒟[:advantage]
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(π, 𝒟[:s]))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
    end 
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

PPO(;π::ActorCritic, ϵ::Float32 = 0.2f0, λp::Float32 = 1f0, λe::Float32 = 0.1f0, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) = 
    OnPolicySolver(;
        π = π,
        𝒫=(ϵ=ϵ, λp=λp, λe=λe),
        log = LoggerParams(;dir = "log/ppo", log...),
        a_opt = TrainingParams(;loss = ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
        c_opt = TrainingParams(;loss = (π, 𝒫, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
        post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
        kwargs...)
    



