function SAC_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        ap, logprob = exploration(actor(π), 𝒟[:sp])
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* (min.(value(π⁻, 𝒟[:sp], ap)...) .- exp(𝒫[:SAC_log_α][1]).*logprob)
    end
end

function SAC_deterministic_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π⁻, 𝒟[:sp], action(actor(π), 𝒟[:sp]))...)
    end
end

function SAC_max_Q_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        error("not implemented")
        #TODO: Sample some number of actions and then choose the max
    end
end

function SAC_actor_loss(π, 𝒫, 𝒟; info = Dict())
    a, logprob = exploration(π.A, 𝒟[:s])
    ignore() do
        info["entropy"] = -mean(logprob)
    end
    mean(exp(𝒫[:SAC_log_α][1]).*logprob .- min.(value(π, 𝒟[:s], a)...))
end

function SAC_temp_loss(π, 𝒫, 𝒟; info = Dict())
    ignore() do
        info["SAC alpha"] = exp(𝒫[:SAC_log_α][1])
    end
    _, logprob = exploration(π.A, 𝒟[:s])
    target_α = logprob .+ 𝒫[:SAC_H_target]
    -mean(exp(𝒫[:SAC_log_α][1]) .* target_α)
end

function SAC(;π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}}, ΔN=50, α::Float32=0.2f0, H_target::Float32 = Float32(-prod(dim(action_space(π)))), π_explore=GaussianNoiseExplorationPolicy(0.1f0), α_opt::NamedTuple=(;), a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) where T
    𝒫 = (SAC_log_α = [Base.log(α)], SAC_H_target = H_target)
    OffPolicySolver(;
        π = π,
        ΔN=ΔN,
        𝒫 = 𝒫,
        log = LoggerParams(;dir = "log/sac", log...),
        param_optimizers = Dict(Flux.params(𝒫[:SAC_log_α]) => TrainingParams(;loss=SAC_temp_loss, name="temp_", α_opt...)),
        a_opt = TrainingParams(;loss=SAC_actor_loss, name="actor_", a_opt...),
        c_opt = TrainingParams(;loss=double_Q_loss, name="critic_", epochs=ΔN, c_opt...),
        π_explore = π_explore,
        target_fn = SAC_target(π),
        kwargs...)
end

