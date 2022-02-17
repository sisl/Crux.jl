function value_estimate(π::DiscreteNetwork, px, s, 𝒫)
        pdfs = logits(px, s)
        sum(value(π, s) .* pdfs, dims=1)
end

function value_estimate(π::ActorCritic, px, s, 𝒫)
        mean([value(π, s, action(px, s)) for _ in 1:𝒫[:N_samples]])
end

#NOTE: Currently all of these assume that we get a reward (cost) ONCE at the end of an episode

function E_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        px = 𝒫[:px]
        if 𝒫[:use_likelihood_weights]
                return 𝒟[:likelihoodweight] .* (𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫))
        else
                return 𝒟[:done] .* 𝒟[:r] .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        end
end

function CDF_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        # rα = 𝒫[:rα][1]
        rs = 𝒫[:rs]
        # stdrα = 𝒫[:std_rα][1]
        # vard = Normal(rα, stdrα)
        px = 𝒫[:px]
        B = length(𝒟[:r])
        
        y = hcat([𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π.policy, px, vcat(repeat([rα], 1, B), 𝒟[:sp]), 𝒫) for rα in rs]...)
        return y
        # return (𝒟[:var_prob] .+ y) ./ 2f0
        # return 𝒟[:var_prob] 
        
        # if 𝒫[:use_likelihood_weights]
        #         return 𝒟[:likelihoodweight] .* (𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫))
        # else
        #          return 𝒟[:done] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        #         # return 𝒟[:done] .* cdf.(vard, 𝒟[:r]) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        # end
end

function CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        rα = 𝒫[:rα][1]
        # stdrα = 𝒫[:std_rα][1]
        # vard = Normal(rα, stdrα)
        px = 𝒫[:px]
        
        y = 𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        return y 
        # return (𝒟[:cvar_prob] .+ y) ./ 2f0
        # return 𝒟[:cvar_prob]
        
        # if 𝒫[:use_likelihood_weights]
        #         return 𝒟[:likelihoodweight] .* (𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫))
        # else
        #         return 𝒟[:done] .* 𝒟[:r] .* (𝒟[:r] .> rα) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        #         # return 𝒟[:done] .* 𝒟[:r] .* cdf.(vard, 𝒟[:r]) .+ (1.f0 .- 𝒟[:done]) .* value_estimate(π, px, 𝒟[:sp], 𝒫)
        # end
end

function E_VaR_CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        [CDF_target(π.networks[1], 𝒫, 𝒟, γ; kwargs...), CVaR_target(π.networks[2], 𝒫, 𝒟, γ; kwargs...), E_target(π.networks[3], 𝒫, 𝒟, γ; kwargs...)]
end 

function VaR_CVaR_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        [CDF_target(π.networks[1], 𝒫, 𝒟, γ; kwargs...), CVaR_target(π.networks[2], 𝒫, 𝒟, γ; kwargs...)]
end

function IS_L_KL_log(π, 𝒫, 𝒟; info=Dict(), kwargs...)
        x, logqx = exploration(π, 𝒟[:s])
        Q_s = Zygote.ignore() do 
                value_estimate(π, 𝒫[:px], 𝒟[:s], 𝒫)
        end
        px = exp.(logpdf(𝒫[:px], 𝒟[:s], x))
        Q_sx = value(π, 𝒟[:s], x)
        
        qstar = px .* Q_sx #./ Q_s
        

        -mean(qstar .* logqx ./ exp.(logqx))
end



function fill_probs(D; 𝒮, info=Dict())
        rα = 𝒮.𝒫[:rα][1]
        epies = !(D isa ExperienceBuffer) ? episodes(ExperienceBuffer(D)) : episodes(D)
        for ep in epies
                episode_range = ep[1]:ep[2]
                r = sum(D[:r][1, episode_range]) # total return
                varprob = r >= rα
                cvarprob = r * varprob
                
                for i in reverse(episode_range)
                        D[:var_prob][:, i] .= varprob
                        D[:cvar_prob][:, i] .= cvarprob
                        
                        varprob = D[:likelihoodweight][1,i] * varprob
                        cvarprob = D[:likelihoodweight][1,i] * cvarprob
                end
        end
end
    






function ISDRL_Discrete(;π,
                        S,
                        N, 
                        px,
                        N_cdf=10,
                        cdf_weights=ones(Float32, N_cdf) ./ N_cdf,
                        prioritized=true,
                        use_likelihood_weights=true, 
                        α,
                        target_fn=VaR_CVaR_target,
                        𝒫=(;),
                        buffer_size=N,
                        ΔN=4,
                        pre_train_callback,
                        π_explore=π, 
                        c_opt::NamedTuple=(;), 
                        log::NamedTuple=(;),
                        c_loss,
                        kwargs...)
               
                    𝒫 = (;px, cdf_weights, rα=Float32[NaN], rs=zeros(Float32, N_cdf), std_rα=Float32[NaN], α, use_likelihood_weights, 𝒫...)
                    required_columns=[:logprob, :likelihoodweight, :var_prob, :cvar_prob]
                    agent = PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π), pa=px)
                    OffPolicySolver(;agent=agent,
                                     S=S,
                                     log=LoggerParams(;dir="log/cerl_dqn", period=100, fns=[log_episode_averages([:r], 100)], log...),
                                     𝒫=𝒫,
                                     N=N,
                                     ΔN=ΔN,
                                     # post_sample_callback=fill_probs,
                                     pre_train_callback=pre_train_callback,
                                     buffer=ExperienceBuffer(S, agent.space, buffer_size, required_columns, prioritized=prioritized),
                                     c_opt = TrainingParams(;loss=c_loss, name="critic_", epochs=ΔN, c_opt...),
                                     target_fn=target_fn,
                                     kwargs...)
end


function ISDRL_Continuous(;π::MixtureNetwork,
                        S,
                        N, 
                        px,
                        N_samples=10,
                        prioritized=true,
                        use_likelihood_weights=true, 
                        α,
                        𝒫=(;),
                        buffer_size=N,
                        ΔN=4,
                        pre_train_callback,
                        π_explore=π, 
                        c_opt::NamedTuple=(;), 
                        a_opt::NamedTuple=(;), 
                        log::NamedTuple=(;),
                        c_loss,
                        a_loss = IS_L_KL_log,
                        kwargs...)
               
                    𝒫 = (;px, rα=[NaN], α, use_likelihood_weights, N_samples, 𝒫...)
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
                                     a_opt = TrainingParams(;loss=a_loss, name="actor_", a_opt...),
                                     target_fn=VaR_CVaR_target,
                                     kwargs...)
end

