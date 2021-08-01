RARL_DQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...) = -𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(value(π, 𝒟[:sp]), dims=1)

RARL(;π::DiscreteNetwork, N::Int, ΔN=4, π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) = 
        AdversarialOffPolicySolver(;
                π=π, 
                log=LoggerParams(;dir="log/dqn", log...),
                N=N,
                ΔN=ΔN,
                c_opt = TrainingParams(;loss=td_loss, name="critic_", epochs=ΔN, c_opt...),
                x_c_opt = TrainingParams(;loss=td_loss, name="x_critic_", epochs=ΔN, c_opt...),
                target_fn=DQN_target,
                x_target_fn=RARL_DQN_target,
                π_explore=π_explore,
                kwargs...)

