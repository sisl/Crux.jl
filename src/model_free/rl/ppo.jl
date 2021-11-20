# PPO loss
function ppo_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a]) 
    r = exp.(new_probs .- 𝒟[:logprob])
    
    A = 𝒟[:advantage]
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(π, 𝒟[:s]))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
    end 
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

function PPO(;π::ActorCritic, 
     ϵ::Float32 = 0.2f0, 
     λp::Float32 = 1f0, 
     λe::Float32 = 0.1f0, 
     target_kl = 0.012f0,
     a_opt::NamedTuple=(;), 
     c_opt::NamedTuple=(;), 
     log::NamedTuple=(;), 
     kwargs...)
     
     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=(ϵ=ϵ, λp=λp, λe=λe),
                    log = LoggerParams(;dir = "log/ppo", log...),
                    a_opt = TrainingParams(;loss = ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = (π, 𝒫, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
                    post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    kwargs...)
end

# PPO loss with a penalty
function lagrange_ppo_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a]) 
    r = exp.(new_probs .- 𝒟[:logprob])
    
    A = 𝒟[:advantage]
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(π, 𝒟[:s]))
    
    #update the cost penalty
    penalty = ignore() do
        # 𝒫[:penalty_param][1] = clamp(𝒫[:penalty_param][1], -7, 10)
        # Flux.softplus(𝒫[:penalty_param][1])
        
        # Average cost
        Jc = sum(𝒟[:cost]) / sum(𝒟[:episode_end])
        # Jc = maximum(𝒟[:cost])
        
        
        # Compute the error
        Δ = Jc - 𝒫[:target_cost]
        
        # Update integral term
        𝒫[:I][1] = max(0, 𝒫[:I][1] + 𝒫[:Ki]*Δ)
        
        # Smooth out the values
        α = 𝒫[:ema_α]
        𝒫[:smooth_Δ][1] = α * 𝒫[:smooth_Δ][1] + (1 - α)*Δ
        𝒫[:smooth_Jc][1] = α * 𝒫[:smooth_Jc][1] + (1 - α)*Jc
        
        # Compute the derivative term
        ∂ = max(0, 𝒫[:smooth_Jc][1] - 𝒫[:Jc_prev][1])
        
        # Update the previous cost
        𝒫[:Jc_prev][1] = 𝒫[:smooth_Jc][1]
        
        # PID update
        penalty = max(0, 𝒫[:Kp] * 𝒫[:smooth_Δ][1] + 𝒫[:I][1] + 𝒫[:Kd]*∂)
        
        info["penalty"] = penalty
        info["cur_cost"] = Jc
        info["smooth_delta"] = 𝒫[:smooth_Δ][1]
        info["deriv_term"] = ∂
        info["Kd"] = 𝒫[:Kd]
        info["Kp"] = 𝒫[:Kp]
        info["integral term"] = 𝒫[:I][1]
        
        
        penalty
    end

    # cost_loss = 𝒫[:penalty_scale] * penalty * mean(r .* 𝒟[:cost_advantage])
    cost_loss = penalty * mean(max.(r .* 𝒟[:cost_advantage], clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* 𝒟[:cost_advantage]))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
    end 
    (𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss + cost_loss) / (1 + penalty)
end

function lagrange_ppo_penalty_loss(π, 𝒫, 𝒟; info = Dict())
    penalty = Flux.softplus(𝒫[:penalty_param][1])
    cur_cost = mean(𝒟[:cost])
    
    ignore() do
        info["penalty"] = penalty
        info["cur_cost"] = cur_cost
    end
    
    -penalty * 𝒫[:penalty_scale] * (cur_cost - 𝒫[:target_cost])
end

function LagrangePPO(;π::ActorCritic,
     Vc::ContinuousNetwork, # value network for estimating cost
     ϵ::Float32 = 0.2f0, 
     λp::Float32 = 1f0, 
     λe::Float32 = 0f0,
     λ_gae = 0.95f0,
     target_kl = 0.012f0,
     penalty_init = 1f0,
     target_cost = 0.025f0,
     penalty_scale = 1f0,
     Ki = 1f-3,
     Kp = 1,
     Kd = 0, 
     ema_α = 0.95,    
     a_opt::NamedTuple=(;), 
     c_opt::NamedTuple=(;), 
     penalty_opt::NamedTuple=(;),
     cost_opt::NamedTuple=(;),
     log::NamedTuple=(;), 
     kwargs...)
     
     𝒫=(ϵ=ϵ, λp=λp, λe=λe, 
        penalty_param=Float32[Base.log(exp(penalty_init)-1)], 
        target_cost=target_cost, 
        penalty_scale=penalty_scale,
        I = [0f0],
        Jc_prev = [0f0],
        Ki=Ki,
        Kp=Kp,
        Kd=Kd,
        ema_α=ema_α,
        smooth_Δ = [0f0],
        smooth_Jc = [0f0]
        )
     
     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=𝒫,
                    Vc=Vc,
                    log = LoggerParams(;dir = "log/lagrange_ppo", log...),
                    # param_optimizers = Dict(Flux.params(𝒫[:penalty_param]) => TrainingParams(;loss=lagrange_ppo_penalty_loss, name="penalty_", penalty_opt...)),
                    a_opt = TrainingParams(;loss = lagrange_ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = (π, 𝒫, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
                    cost_opt = TrainingParams(;loss = (π, 𝒫, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:cost_return]), name = "cost_critic_", cost_opt...),
                    required_columns = [:return, :advantage, :logprob, :cost_advantage, :cost, :cost_return],
                    post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    kwargs...)
end




        
    



