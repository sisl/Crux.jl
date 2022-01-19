function discrete_value_estimate(π, px, s)
        pdfs = exp.(logpdf(px, s))
        sum(value(π, s) .* pdfs, dims=1)
end

function CERL_double_loss(;name1=:Q1avg, name2=:Q2avg, kwargs...)
    l1 = td_loss(;name=name1, kwargs...)
    l2 = td_loss(;name=name2, kwargs...)
    
    (π, 𝒫, 𝒟, y; info=Dict()) -> begin
        .5f0*(l1(π.C.N1, 𝒫, 𝒟, y[1], info=info) + l2(π.C.N2, 𝒫, 𝒟, y[2], info=info))
    end
end

#NOTE: Currently all of these assume that we get a reward (cost) ONCE at the end of an episode

function expected_reward_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        px = 𝒫[:px]
        𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* discrete_value_estimate(π, px, 𝒟[:sp])
end

function expected_tail_reward_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        rα = 𝒫[:rα]
        px = 𝒫[:px]
        𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* discrete_value_estimate(π.C.N1, px, 𝒟[:sp]) #CDF
        𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* discrete_value_estimate(π.C.N2, px, 𝒟[:sp]) #CVAR
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
                        c_loss=CERL_double_loss(),
                        kwargs...)
               
                    𝒫 = (;px, rα=NaN, 𝒫...)
                    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
                                     log=LoggerParams(;dir="log/cerl_dqn", log...),
                                     𝒫=𝒫,
                                     N=N,
                                     ΔN=ΔN,
                                     buffer_size=buffer_size,
                                     c_opt = TrainingParams(;loss=c_loss, name="critic_", epochs=ΔN, c_opt...),
                                     target_fn=expected_tail_reward_target,
                                     kwargs...)
end

