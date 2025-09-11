"""
TD3 target function.
"""
function td3_target(π, 𝒫, 𝒟, γ::Float32; i)
    ap, _ = exploration(𝒫[:π_smooth], 𝒟[:sp], π_on=π, i=i)
    y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π, 𝒟[:sp], ap)...)
end

"""
TD3 actor loss function.
"""
td3_actor_loss(π, 𝒫, 𝒟; info = Dict()) = -mean(value(π.C.N1, 𝒟[:s], action(π, 𝒟[:s])))

"""
Twin Delayed DDPG (TD3) solver.

```julia
TD3(;
    π,
    ΔN=50,
    π_smooth::Policy=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0),
    π_explore=GaussianNoiseExplorationPolicy(0.1f0),
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    a_loss=td3_actor_loss,
    c_loss=double_Q_loss(),
    target_fn=td3_target,
    prefix="",
    log::NamedTuple=(;),
    𝒫::NamedTuple=(;),
    kwargs...)
```
"""
function TD3(;
        π,
        ΔN=50,
        π_smooth::Policy=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0),
        π_explore=GaussianNoiseExplorationPolicy(0.1f0),
        a_opt::NamedTuple=(;),
        c_opt::NamedTuple=(;),
        a_loss=td3_actor_loss,
        c_loss=double_Q_loss(),
        target_fn=td3_target,
        prefix="",
        log::NamedTuple=(;),
        𝒫::NamedTuple=(;),
        kwargs...)

    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)),
                     ΔN=ΔN,
                     𝒫=(π_smooth=π_smooth, 𝒫...),
                     log=LoggerParams(;dir = "log/td3", log...),
                     a_opt=TrainingParams(;loss=a_loss, name=string(prefix, "actor_"), a_opt...),
                     c_opt=TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                     target_fn=target_fn,
                     kwargs...)
end
