function a2c_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    p_loss = -mean(new_probs .* 𝒟[:advantage])
    e_loss = -mean(entropy(π, 𝒟[:s]))
    
    # Log useful information
    ignore() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
    end 
    
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

function A2C(;π::ActorCritic, 
              a_opt::NamedTuple=(;), 
              c_opt::NamedTuple=(;), 
              log::NamedTuple=(;), 
              λp::Float32=1f0, 
              λe::Float32=0.1f0, kwargs...)
              
    OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=(λp=λp, λe=λe),
                    log=LoggerParams(;dir = "log/a2c", log...),
                    a_opt=TrainingParams(;loss=a2c_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
                    c_opt=TrainingParams(;loss=(π, 𝒫, D; kwargs...) -> Flux.mse(value(π, D[:s]), D[:return]), name = "critic_", c_opt...),
                    post_batch_callback=(𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    kwargs...)
end
        
    



