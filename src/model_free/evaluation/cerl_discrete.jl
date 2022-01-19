function discrete_value_estimate(π, px, s)
        pdfs = exp.(logpdf(px, s))
        sum(value(π, s) .* pdfs, dims=1)
end

function CERL_Discrete_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        px = 𝒫[:px]
        𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* discrete_value_estimate(π, px, 𝒟[:sp])
end

function CERL_Discrete(;π::DiscreteNetwork,
                        N, 
                        px,
                        𝒫=(;),
                        buffer_size=N,
                        ΔN=4,
                        π_explore, 
                        c_opt::NamedTuple=(;), 
                        log::NamedTuple=(;),
                        c_loss=td_loss(),
                        kwargs...)
               
                    𝒫 = (;px, 𝒫...)
                    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
                                     log=LoggerParams(;dir="log/cerl_dqn", log...),
                                     𝒫=𝒫,
                                     N=N,
                                     ΔN=ΔN,
                                     buffer_size=buffer_size,
                                     c_opt = TrainingParams(;loss=c_loss, name="critic_", epochs=ΔN, c_opt...),
                                     target_fn=CERL_Discrete_target,
                                     kwargs...)
end

