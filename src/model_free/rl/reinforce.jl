# REINFORCE loss
reinforce_loss() = (π, 𝒟; info = Dict()) -> reinforce_loss(π, 𝒟[:s], 𝒟[:a], 𝒟[:return], 𝒟[:logprob], info)
function reinforce_loss(π, s, a, G, old_probs, info = Dict())
    new_probs = logpdf(π, s, a)
    
    ignore() do
        info[:entropy] = mean(entropy(π, s))
        info[:kl] = mean(old_probs .- new_probs)
    end 
    
    -mean(new_probs .* G)
end

# Build a REINFORCE solver
REINFORCE(;a_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...) = 
    OnPolicySolver(;
        log = LoggerParams(;dir = "log/reinforce", log...),
        a_opt = TrainingParams(;loss = reinforce_loss(), early_stopping = (info) -> (info[:kl] > 0.015), name = "actor_", a_opt...),
        kwargs...)
    



