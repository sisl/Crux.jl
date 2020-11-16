struct PPOSolver <: Solver 


end

function POMDPS.solve(𝒮::PPOSolver, mdp)
    for i=1:N
        # Store the previous policy so that the KL-Divergence can be computed between steps
        prev_policy = deepcopy(policy)

        # Compute the gradients and update the parameters
        grads = gradient(() -> lossfn(policy), params)
        update_with_clip!(opt, grads, params, max_norm)

    end
end

function ppo_batch_loss(task, policy, N_eps, γ, λ, baseline_reg_coeff; ϵ = 0.2, baseline_weights = nothing, store_log = false, logger = nothing)
    # Sample a batch from the current policy
    batch = sample_batch(task, policy, N_eps)
    store_log && add_entry(logger, "return", average_episode_return(batch))
    store_log && add_entry(logger, "max_return", max_episode_return(batch))
    store_log && (logger["last_obs"] = batch.observations)

    # Compute advantages with linear baseline
    isnothing(baseline_weights) && (baseline_weights = fit(batch, baseline_reg_coeff))
    advantages = gae(batch, baseline_weights, γ, λ)

    # Compute the log ratio fro the derivative and use it for the loss
    old_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...).data
    new_log_probs = log_prob(batch.actions, forward_nn(policy, batch.observations)...)
    ratio = exp.(new_log_probs .- old_log_probs)
    loss = -mean(min.(ratio.*advantages, clamp.(ratio, 1-ϵ, 1+ϵ).*advantages))
    store_log && add_entry(logger, "loss", loss)
    loss
end

# Generic training loop that applies clipping and stores training info
function train!(policy, lossfn, N, lr, max_norm, training_log, policy_filename = nothing)

    # Pull out the vector of parameters for Flux
    params = to_params(policy)

end

# Clips the norm of the gradient to max norm
# Returns the scaling factor to multiply the gradient to
function clip_norm_scale(grads, params, max_norm, p=2)
    gnorm = grad_norms(grads, params, p)
    gnorm > max_norm ? max_norm/gnorm : 1
end

# Returns the clipped value of the gradient norms
# used for diagnosing
clipped_grad_norms(grads, params, max_norm, p=2) = min(max_norm, grad_norms(grads, params, p))

# update the parameters with gradient clipping
function update_with_clip!(opt, grads, params, max_norm, p=2)
    scale = clip_norm_scale(grads, params, max_norm, p)
    for p in params
        Flux.Tracker.update!(opt, p, scale*grads[p])
    end
end

function features(batch::Batch)
    hcat(batch.observations, batch.observations .^ 2, batch.times, batch.times .^ 2, batch.times .^ 3, ones(batch.N))
end


