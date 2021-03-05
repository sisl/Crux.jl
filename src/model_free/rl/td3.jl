function TD3_target(π_smooth)
    (π, 𝒟, γ::Float32; i) -> begin
        ap, _ = exploration(π_smooth, 𝒟[:sp], π_on=π, i=i)    
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π, 𝒟[:sp], ap)...)
    end
end

TD3_actor_loss(π, 𝒟; info = Dict()) = -mean(value(π.C.N1, 𝒟[:s], action(π, 𝒟[:s])))

TD3(;π::ActorCritic{ContinuousNetwork, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}}, ΔN=50, 
     π_smooth::Policy=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), π_explore=GaussianNoiseExplorationPolicy(0.1f0), a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) = 
    OffPolicySolver(;
        π = π,
        ΔN=ΔN,
        log = LoggerParams(;dir = "log/td3", log...),
        a_opt = TrainingParams(;loss=TD3_actor_loss, name="actor_", update_every=2, a_opt...),
        c_opt = TrainingParams(;loss=double_Q_loss, name="critic_", epochs=ΔN, c_opt...),
        π_explore = π_explore,
        target_fn = TD3_target(π_smooth),
        kwargs...)

