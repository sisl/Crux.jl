function value_estimate(π::DiscreteNetwork, px, s) where {T <: DiscreteNetwork}
        pdfs = exp.(logpdf(px, s))
        sum(value(π, s) .* pdfs, dims=1)
end

#NOTE: Currently all of these assume that we get a reward (cost) ONCE at the end of an episode

function E_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        px = 𝒫[:px]
        if 𝒫[:use_likelihood_weights]
                return 𝒟[:likelihoodweight] .* (𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp]))
        else
                return 𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp])
        end
end

function CDF_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        rα = 𝒫[:rα][1]
        px = 𝒫[:px]
        
        if 𝒫[:use_likelihood_weights]
                return 𝒟[:likelihoodweight] .* (𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp]))
        else
                return 𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp])
        end
end

function CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        rα = 𝒫[:rα][1]
        px = 𝒫[:px]
        if 𝒫[:use_likelihood_weights]
                return 𝒟[:likelihoodweight] .* (𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp]))
        else
                return 𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp])
        end
end

function E_VaR_CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        [CDF_target(π.networks[1], 𝒫, 𝒟, γ; kwargs...), CVaR_target(π.networks[2], 𝒫, 𝒟, γ; kwargs...), E_target(π.networks[3], 𝒫, 𝒟, γ; kwargs...)]
end 

function VaR_CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        [CDF_target(π.networks[1], 𝒫, 𝒟, γ; kwargs...), CVaR_target(π.networks[2], 𝒫, 𝒟, γ; kwargs...)]
end


function CERL_Discrete(;π::MixtureNetwork,
                        S,
                        N, 
                        px,
                        prioritized=true,
                        use_likelihood_weights=true, 
                        α,
                        𝒫=(;),
                        buffer_size=N,
                        ΔN=4,
                        pre_train_callback,
                        π_explore, 
                        c_opt::NamedTuple=(;), 
                        log::NamedTuple=(;),
                        c_loss,
                        kwargs...)
               
                    𝒫 = (;px, rα=[NaN], α, use_likelihood_weights, 𝒫...)
                    required_columns=[:logprob, :likelihoodweight]
                    agent = PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π), pa=px)
                    OffPolicySolver(;agent=agent,
                                     S=S,
                                     log=LoggerParams(;dir="log/cerl_dqn", period=100, fns=[log_episode_averages([:r], 100)], log...),
                                     𝒫=𝒫,
                                     N=N,
                                     ΔN=ΔN,
                                     pre_train_callback=pre_train_callback,
                                     buffer=ExperienceBuffer(S, agent.space, buffer_size, required_columns, prioritized=prioritized),
                                     c_opt = TrainingParams(;loss=c_loss, name="critic_", epochs=ΔN, c_opt...),
                                     target_fn=VaR_CVaR_target,
                                     kwargs...)
end

