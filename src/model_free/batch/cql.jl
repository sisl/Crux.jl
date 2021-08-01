function CQL_alpha_loss(π, 𝒫, 𝒟; info = Dict())
    ignore() do
        info["CQL alpha"] = exp(𝒫[:CQL_log_α][1])
    end
    -conservative_loss(π, 𝒫, 𝒟)
end

function importance_sampling(πsamp, π, obs, Nsamples)
    @assert ndims(obs) == 2 # does not support multidimensional observations yet
    @assert critic(π) isa DoubleNetwork # Assumes we have a double network
    
    rep_obs, flat_actions, logprobs = Zygote.ignore() do
        actions_and_logprobs = [exploration(πsamp, obs) for i=1:Nsamples]
        actions = cat([a for (a, _) in actions_and_logprobs]..., dims=3)
        logprobs = cat([lp for (_, lp) in actions_and_logprobs]..., dims=3)
        rep_obs = repeat(obs, 1, Nsamples)
        flat_actions = reshape(actions, size(actions)[1], :)
        rep_obs, flat_actions, logprobs
    end
    
    qvals = reshape(mean(value(π, rep_obs, flat_actions)), 1, :, Nsamples)
    
    return qvals .- logprobs
end

function conservative_loss(π, 𝒫, 𝒟; info=Dict())
    obs = 𝒟[:s]
    acts = 𝒟[:a]
    pol_values = importance_sampling(π, π, obs, 𝒫[:CQL_n_action_samples])
    unif_values = importance_sampling(𝒫[:CQL_is_distribution], π, obs, 𝒫[:CQL_n_action_samples])
    combined = cat(pol_values, unif_values, dims=3)
    lse = logsumexp(combined, dims=3)
    loss = mean(lse) - mean(mean(value(π, obs, acts)))
    
    β = clamp(exp(𝒫[:CQL_log_α][1]), 0f0, 1f6)
    β * (5f0*loss - 𝒫[:CQL_α_thresh])
end

function CQL_critic_loss(π, 𝒫, 𝒟, y; info=Dict(), weighted=false)
    loss = double_Q_loss(π, 𝒫, 𝒟, y, info=info)
    c_loss = conservative_loss(π, 𝒫, 𝒟, info=info)
    loss + c_loss
end

function CQL(;π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
    solver_type = BatchSAC,
    CQL_α::Float32=1f0,
    CQL_is_distribution = DistributionPolicy(product_distribution([Uniform(-1,1) for i=1:dim(action_space(π))[1]])),
    CQL_α_thresh::Float32 = 10f0,
    CQL_n_action_samples::Int = 10,
    CQL_α_opt::NamedTuple=(;),
    a_opt::NamedTuple=(;), 
    c_opt::NamedTuple=(;), 
    log::NamedTuple=(;),
    kwargs...) where T
    # Fill the parameters
    𝒫 = (CQL_log_α=[Base.log(CQL_α)],
          CQL_is_distribution=CQL_is_distribution,
          CQL_n_action_samples=CQL_n_action_samples,
          CQL_α_thresh=CQL_α_thresh)
    solver_type(;
        π = π,
        𝒫 = 𝒫,
        log = (;dir = "log/cql", log...),
        param_optimizers = Dict(Flux.params(𝒫[:CQL_log_α]) => TrainingParams(;loss=CQL_alpha_loss, name="CQL_alpha_", CQL_α_opt...)),
        a_opt = a_opt,
        c_opt = (loss=CQL_critic_loss, name="critic_", c_opt...),
        kwargs...)
end

