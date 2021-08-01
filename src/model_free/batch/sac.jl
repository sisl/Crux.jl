function BatchSAC(;π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}}, S, A=action_space(π), ΔN=50, SAC_α::Float32=1f0, SAC_H_target::Float32 = Float32(-prod(dim(action_space(π)))), 𝒟_train, SAC_α_opt::NamedTuple=(;), a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), 𝒫::NamedTuple=(;), param_optimizers=Dict(), normalize_training_data = true, kwargs...) where T
    normalize_training_data && (𝒟_train = normalize!(deepcopy(𝒟_train), S, A))
    𝒟_train = 𝒟_train |> device(π)
    
    𝒫 = (SAC_log_α = [Base.log(SAC_α)], SAC_H_target = SAC_H_target, 𝒫...)
    BatchSolver(;
        π = π,
        S=S,
        A=A,
        𝒫 = 𝒫,
        𝒟_train=𝒟_train,
        log = LoggerParams(;dir = "log/batch_sac", log...),
        param_optimizers = Dict(Flux.params(𝒫[:SAC_log_α]) => TrainingParams(;loss=SAC_temp_loss, name="temp_", SAC_α_opt...), param_optimizers...),
        a_opt = TrainingParams(;loss=SAC_actor_loss, name="actor_", a_opt...),
        c_opt = TrainingParams(;loss=double_Q_loss, name="critic_", epochs=ΔN, c_opt...),
        target_fn = SAC_target(π),
        kwargs...)
end

