# REINFORCE loss
function reinforce_loss(π, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(π, 𝒟[:s], 𝒟[:a])
    
    ignore() do
        info[:entropy] = mean(entropy(π, 𝒟[:s]))
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
    end 
    
    -mean(new_probs .* 𝒟[:return])
end

# Build a REINFORCE solver
function REINFORCE(;π,
                    a_opt::NamedTuple=(;), 
                    log::NamedTuple=(;),
                    required_columns=[],
                    kwargs...)
                    
    OnPolicySolver(;agent=PolicyParams(π),
                    log = LoggerParams(;dir = "log/reinforce", log...),
                    a_opt = TrainingParams(;loss = reinforce_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
                    required_columns = unique([required_columns..., :return, :logprob]),
                    kwargs...)
end
        
    



