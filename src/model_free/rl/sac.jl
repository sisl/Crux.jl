"""
SAC target function.
"""
function sac_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        ap, logprob = exploration(actor(π), 𝒟[:sp])
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* (min.(value(π⁻, 𝒟[:sp], ap)...) .- exp(𝒫[:SAC_log_α][1]).*logprob)
    end
end

"""
Deterministic SAC target function.
"""
function sac_deterministic_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π⁻, 𝒟[:sp], action(actor(π), 𝒟[:sp]))...)
    end
end

"""
Max-Q SAC target function.
"""
function sac_max_q_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        error("not implemented")
        #TODO: Sample some number of actions and then choose the max
    end
end


"""
SAC actor loss function.
"""
function sac_actor_loss(π, 𝒫, 𝒟; info = Dict())
    a, logprob = exploration(π.A, 𝒟[:s])
    ignore_derivatives() do
        info["entropy"] = -mean(logprob)
    end
    mean(exp(𝒫[:SAC_log_α][1]).*logprob .- min.(value(π, 𝒟[:s], a)...))
end

"""
SAC temp-based loss function.
"""
function sac_temp_loss(π, 𝒫, 𝒟; info = Dict())
    ignore_derivatives() do
        info["SAC alpha"] = exp(𝒫[:SAC_log_α][1])
    end
    _, logprob = exploration(π.A, 𝒟[:s])
    target_α = logprob .+ 𝒫[:SAC_H_target]
    -mean(exp(𝒫[:SAC_log_α][1]) .* target_α)
end


"""
Soft Actor Critic (SAC) solver.

```julia
SAC(;
    π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
    ΔN=50,
    SAC_α::Float32=1f0,
    SAC_H_target::Float32 = Float32(-prod(dim(action_space(π)))),
    π_explore=GaussianNoiseExplorationPolicy(0.1f0),
    SAC_α_opt::NamedTuple=(;),
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    a_loss=sac_actor_loss,
    c_loss=double_Q_loss(),
    target_fn=sac_target(π),
    prefix="",
    log::NamedTuple=(;),
    𝒫::NamedTuple=(;),
    param_optimizers=Dict(),
    kwargs...)
```
"""
function SAC(;
        π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
        ΔN=50,
        SAC_α::Float32=1f0,
        SAC_H_target::Float32 = Float32(-prod(dim(action_space(π)))),
        π_explore=GaussianNoiseExplorationPolicy(0.1f0),
        SAC_α_opt::NamedTuple=(;),
        a_opt::NamedTuple=(;),
        c_opt::NamedTuple=(;),
        a_loss=sac_actor_loss,
        c_loss=double_Q_loss(),
        target_fn=sac_target(π),
        prefix="",
        log::NamedTuple=(;),
        𝒫::NamedTuple=(;),
        param_optimizers=Dict(),
        kwargs...) where T

    𝒫 = (SAC_log_α=[Base.log(SAC_α)], SAC_H_target=SAC_H_target, 𝒫...)
    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)),
                     ΔN=ΔN,
                     𝒫=𝒫,
                     log=LoggerParams(;dir = "log/sac", log...),
                     param_optimizers=Dict(Flux.params(𝒫[:SAC_log_α]) => TrainingParams(;loss=sac_temp_loss, name="temp_", SAC_α_opt...), param_optimizers...),
                     a_opt=TrainingParams(;loss=a_loss, name=string(prefix, "actor_"), a_opt...),
                     c_opt=TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                     target_fn=target_fn,
                     kwargs...)
end
