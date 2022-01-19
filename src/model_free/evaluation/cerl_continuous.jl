# function discrete_value_estimate(π, px, s)
#         xs = support(px)
#         pdfs = exp.(logpdf(px, s, xs))
#         sum(value(π, s) .* pdfs, dims=1)
# end
# 
# function CERL_DQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
#         px = 𝒫[:px]
#         term = 𝒟[:done] .| 𝒟[:fail]
#         term .* 𝒟[:fail] .+ (1.f0 .- term) .* (discrete_value_estimate(π, px, 𝒟[:sp]) ./ discrete_value_estimate(π, px, 𝒟[:s]))
# end
# 
# function IS_estimate_log_pfail(π, px, s, Nsamples=10)
#         xs_and_logpdfs = [exploration(π, s) for _ in 1:Nsamples]
#         xs = [e[1] for e in xs_and_logpdfs]
#         νlogpdfs = vcat([e[2] for e in xs_and_logpdfs]...)
#         logpfails = vcat([value(π, s, x) for x in xs]...)
#         logpxs = Float32.(vcat([logpdf.(px, x) for x in xs]...))
# 
#         -Float32(Base.log(Nsamples)) .+ logsumexp(νlogpdfs .+ logpfails .- logpxs, dims=1)
# end
# 
# function estimate_log_pfail(π, px, s, Nsamples=10)
#         xs = [reshape(rand(px, size(s,2)), :, size(s,2)) for _ in 1:Nsamples]
#         logpfails = vcat([value(π, s, x) for x in xs]...)
# 
#         -Float32(Base.log(Nsamples)) .+ logsumexp(logpfails, dims=1)
# end
# 
# function estimate_pfail(π, px, s, Nsamples=10)
#         xs = [reshape(rand(px, size(s,2)), :, size(s,2)) for _ in 1:Nsamples]
#         pfails = vcat([value(π, s, x) for x in xs]...)
# 
#         mean(pfails, dims=1)
# end
# 
# function IS_Continuous_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
#         term = 𝒟[:done] .| 𝒟[:fail]
#         # term .* (𝒟[:fail] .== false) .* -100f0 .+ (1.f0 .- term) .* estimate_log_pfail(π, 𝒫[:px], 𝒟[:sp], 𝒫[:N_IS_Samples])
#         norm = estimate_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
#         term .* 𝒟[:fail] .+ (1.f0 .- term) .* (estimate_pfail(π, 𝒫[:px], 𝒟[:sp], 𝒫[:N_IS_Samples])) ./ norm
# end
# 
# function IS_L_KL_log(π, 𝒫, 𝒟; kwargs...)
#         x, logνx = exploration(π, 𝒟[:s])
#         logpfail_s = Zygote.ignore() do 
#                 estimate_log_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
#         end
#         logpx = logpdf(𝒫[:px], x)
#         logpfail_sx = value(π, 𝒟[:s], x)
# 
# 
#         -mean(exp.( logpx .+ logpfail_sx .- logνx .- logpfail_s) .* logνx)
# end
# 
# function IS_L_KL(π, 𝒫, 𝒟; kwargs...)
#         x, logνx = exploration(π, 𝒟[:s])
#         νx = exp.(logνx)
#         pfail_s = Zygote.ignore() do 
#                 estimate_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
#         end
#         px = exp.(logpdf(𝒫[:px], x))
#         pfail_sx = value(π, 𝒟[:s], x)
# 
# 
#         -mean(logνx .* px .* pfail_sx ./ (νx .* pfail_s))
# end
# 
# 
# function compute_IS_weight(𝒟, 𝒫; info=Dict())
#         𝒟[:weight] .= exp.(sum(𝒫[:xlogprobs] .* 𝒟[:x], dims=1) .- 𝒟[:xlogprob])
#         info[:mean_is_weight] = mean(𝒟[:weight])
#         info[:std_is_weight] = std(𝒟[:weight])
#         info[:min_is_weight] = minimum(𝒟[:weight])
#         info[:max_is_weight] = maximum(𝒟[:weight])
# end
# 
# function compute_IS_weight_continuous(𝒟, 𝒫; info=Dict())
#         if 𝒫[:px] isa UnivariateDistribution
#                 𝒟[:weight] .= clamp.(exp.(reshape(logpdf.(𝒫[:px], 𝒟[:x]), 1, :) .- 𝒟[:xlogprob]), 0f0, 5f0)
#         else
#                 𝒟[:weight] .= clamp.(exp.(reshape(logpdf(𝒫[:px], 𝒟[:x]), 1, :) .- 𝒟[:xlogprob]), 0f0, 5f0)
#         end
#         info[:mean_is_weight] = mean(𝒟[:weight])
#         info[:std_is_weight] = std(𝒟[:weight])
#         info[:min_is_weight] = minimum(𝒟[:weight])
#         info[:max_is_weight] = maximum(𝒟[:weight])
# end
# 
# ISARL_DQN(;kwargs...) = DQN(;c_loss=td_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=IS_DQN_target)
# ISARL_DDPG(;kwargs...) = DQN(;a_loss=IS_L_KL, c_loss=td_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=IS_Continuous_target, )
# 
# function CERL_DQN(;π::DiscreteNetwork, 
#               N::Int, 
#               ΔN=4, 
#               π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs), 
#               c_opt::NamedTuple=(;), 
#               log::NamedTuple=(;),
#               c_loss=td_loss(),
#               target_fn=DQN_target,
#               prefix="",
#               kwargs...)
# 
# 
#                     OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
#                                      log=LoggerParams(;dir="log/cerl_dqn", log...),
#                                      N=N,
#                                      ΔN=ΔN,
#                                      c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
#                                      target_fn=target_fn,
#                                      kwargs...)
# end