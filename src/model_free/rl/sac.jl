function SAC_target(π, α)
    (π⁻, 𝒟, γ; kwargs...) -> begin
        ap, logprob = exploration(π.A, 𝒟[:sp])
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* (min.(value(π⁻, 𝒟[:sp], ap)...) .- α*logprob)
    end
end

function SAC_actor_loss(α)
    (π, 𝒟; info = Dict()) -> begin
        a, logprob = exploration(π.A, 𝒟[:s])
        mean(α*logprob .- min.(value(π, 𝒟[:s], a)...))
    end
end

SAC(;π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}}, ΔN=50, α::Float32=0.2f0, π_explore=GaussianNoiseExplorationPolicy(0.1f0), a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) where T = 
    OffPolicySolver(;
        π = π,
        ΔN=ΔN,
        log = LoggerParams(;dir = "log/sac", log...),
        a_opt = TrainingParams(;loss=SAC_actor_loss(α), name="actor_", a_opt...),
        c_opt = TrainingParams(;loss=double_Q_loss, name="critic_", epochs=ΔN, c_opt...),
        π_explore = π_explore,
        target_fn = SAC_target(π, α),
        kwargs...)

