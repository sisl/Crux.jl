function IS_DQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        term = 𝒟[:done] .| 𝒟[:fail]
        term .* Base.log.(𝒟[:fail] .+ 1f-16) .+ (1.f0 .- term) .* logsumexp(𝒫[:xlogprobs] .+ value(π, 𝒟[:sp]), dims=1)
        # abs.(𝒟[:r]) .* (𝒟[:r] .< -10)  .+ (1.f0 .- 𝒟[:done]) .* γ .* sum(exp.(𝒫[:xlogprobs]) .* value(π, 𝒟[:sp]), dims=1)
        # 𝒟[:done] .* 𝒟[:fail] .+ (1.f0 .- 𝒟[:done]) .* sum(exp.(𝒫[:xlogprobs]) .* value(π, 𝒟[:sp]), dims=1)
end

function IS_estimate_log_pfail(π, px, s, Nsamples=10)
        xs_and_logpdfs = [exploration(π, s) for _ in 1:Nsamples]
        xs = [e[1] for e in xs_and_logpdfs]
        νlogpdfs = vcat([e[2] for e in xs_and_logpdfs]...)
        logpfails = vcat([value(π, s, x) for x in xs]...)
        logpxs = Float32.(vcat([logpdf.(px, x) for x in xs]...))

        -Float32(Base.log(Nsamples)) .+ logsumexp(νlogpdfs .+ logpfails .- logpxs, dims=1)
end

function estimate_log_pfail(π, px, s, Nsamples=10)
        xs = [reshape(rand(px, size(s,2)), :, size(s,2)) for _ in 1:Nsamples]
        logpfails = vcat([value(π, s, x) for x in xs]...)

        -Float32(Base.log(Nsamples)) .+ logsumexp(logpfails, dims=1)
end

function estimate_pfail(π, px, s, Nsamples=10)
        xs = [reshape(rand(px, size(s,2)), :, size(s,2)) for _ in 1:Nsamples]
        pfails = vcat([value(π, s, x) for x in xs]...)
        
        mean(pfails, dims=1)
end

function IS_Continuous_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        term = 𝒟[:done] .| 𝒟[:fail]
        # term .* (𝒟[:fail] .== false) .* -100f0 .+ (1.f0 .- term) .* estimate_log_pfail(π, 𝒫[:px], 𝒟[:sp], 𝒫[:N_IS_Samples])
        term .* 𝒟[:fail] .+ (1.f0 .- term) .* estimate_pfail(π, 𝒫[:px], 𝒟[:sp], 𝒫[:N_IS_Samples])
end

function IS_L_KL_log(π, 𝒫, 𝒟; kwargs...)
        x, logνx = exploration(π, 𝒟[:s])
        logpfail_s = Zygote.ignore() do 
                estimate_log_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
        end
        logpx = logpdf(𝒫[:px], x)
        logpfail_sx = value(π, 𝒟[:s], x)
        
        
        -mean(exp.( logpx .+ logpfail_sx .- logνx .- logpfail_s) .* logνx)
end

function IS_L_KL(π, 𝒫, 𝒟; kwargs...)
        x, logνx = exploration(π, 𝒟[:s])
        νx = exp.(logνx)
        pfail_s = Zygote.ignore() do 
                estimate_pfail(π, 𝒫[:px], 𝒟[:s], 𝒫[:N_IS_Samples])
        end
        px = exp.(logpdf(𝒫[:px], x))
        pfail_sx = value(π, 𝒟[:s], x)
        
        
        -mean(logνx .* px .* pfail_sx ./ (νx .* pfail_s))
end


function compute_IS_weight(𝒟, 𝒫; info=Dict())
        𝒟[:weight] .= exp.(sum(𝒫[:xlogprobs] .* 𝒟[:x], dims=1) .- 𝒟[:xlogprob])
        info[:mean_is_weight] = mean(𝒟[:weight])
        info[:std_is_weight] = std(𝒟[:weight])
        info[:min_is_weight] = minimum(𝒟[:weight])
        info[:max_is_weight] = maximum(𝒟[:weight])
end

function compute_IS_weight_continuous(𝒟, 𝒫; info=Dict())
        if 𝒫[:px] isa UnivariateDistribution
                𝒟[:weight] .= clamp.(exp.(reshape(logpdf.(𝒫[:px], 𝒟[:x]), 1, :) .- 𝒟[:xlogprob]), 0f0, 5f0)
        else
                𝒟[:weight] .= clamp.(exp.(reshape(logpdf(𝒫[:px], 𝒟[:x]), 1, :) .- 𝒟[:xlogprob]), 0f0, 5f0)
        end
        info[:mean_is_weight] = mean(𝒟[:weight])
        info[:std_is_weight] = std(𝒟[:weight])
        info[:min_is_weight] = minimum(𝒟[:weight])
        info[:max_is_weight] = maximum(𝒟[:weight])
end

ISARL_DQN(;kwargs...) = DQN(;c_loss=td_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=IS_DQN_target)
ISARL_DDPG(;kwargs...) = DQN(;a_loss=IS_L_KL, c_loss=td_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=IS_Continuous_target, )

function ISARL(;𝒮_pro,
               𝒮_ant,
               px,
               log::NamedTuple=(;), 
               train_pro_every::Int=1,
               train_ant_every::Int=1,
               buffer_size=1000, # Size of the buffer
               required_columns=[:fail, :x, :xlogprob, :weight],
               buffer::ExperienceBuffer=ExperienceBuffer(𝒮_pro.S, 𝒮_pro.agent.space, buffer_size, required_columns), # The replay buffer
               buffer_init::Int = max(max(𝒮_pro.c_opt.batch_size, 𝒮_ant.c_opt.batch_size), 200) # Number of observations to initialize the buffer with
               )
               
               # Set buffer parameters to be consistent between solvers (Since buffer is shared)
       𝒮_pro.required_columns = required_columns
       𝒮_pro.buffer = buffer
       𝒮_pro.buffer_init = buffer_init
       𝒮_ant.required_columns = required_columns
       𝒮_ant.buffer = buffer    
       𝒮_ant.buffer_init = buffer_init     
        AdversarialOffPolicySolver(;𝒮_pro=𝒮_pro,
                                    𝒮_ant=𝒮_ant,
                                    px=px,
                                    train_pro_every=train_pro_every,
                                    train_ant_every=train_ant_every,
                                    log=LoggerParams(;dir="log/isarl", log...),)
end