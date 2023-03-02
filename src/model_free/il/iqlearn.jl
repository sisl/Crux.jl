# iqlearn off-policy, keep track of buffer that isnt thrown away 

# concat 50% expert and 50% training batch
# V, V' = getV(s), getV(s')
# Q = Q(s,a)
#
# iq_loss(pi, Q, V, V') = 1) soft_q_loss + 2) value_loss
#
# y = (1-dones)*γ*V'
# r = Q-y
# 1) r_expert =r[expert_indices]
# with no_grad(): phi_grad = f(r_expert) <- 1 for default
# soft_q = -(phi_grad*r_expert).mean()
#
# online: 2) value_loss = (V-y).mean()
# offline: 2) value_loss = (V-y)[expert_indices].mean()
#
# Other tricks to add to loss:
# 
# Grad penalty:
# Interpolate between expert and demo states
# compute 2-norm of jacobian of those interpolated states 
# grad_penalty = lambda * ((gradients_norm - 1) ** 2).mean()
# 
# χ² divergence (offline):
# chi2_loss = 1/(4α)* (r**2)[expert_indices].mean()
#
#
# χ² regularization (online):
# reg_loss = 1/(4α)* (r**2).mean()

# actor lr = 3e-5
# actor init temp = 
# critic_target_update_frequency: 4
# critic_lr: 1e-4
# critic_betas: [0.9, 0.999]
# init_temp: 0.01
# critic_target_update_frequency: 4
# critic_tau: 0.1
# method
# loss div: -
# alpha: 0.5
# lambda_gp: 10

function IQ_loss(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    V = soft_value(π, 𝒟[:s])
    Vp = soft_value(π, 𝒟[:sp])
    Q = value(π, 𝒟[:s], onehotbatch(𝒟[:a]))
    y = γ .* (1.f0 .- 𝒟[:done]) .* Vp
    R = Q-y

    p1 = mean(-R[expert])
    p2 = mean(V-y)

    loss = p1+p2
    if gradient_penalty
        gp = ...
        loss += gp
    end
    if regularize
        reg_loss = 1/(4*α) .* mean(R .^ 2)
        loss += reg_loss
    end
    loss
end

function OnlineIQLearn(;π, 
    S, 
    𝒟_demo, 
    normalize_demo::Bool=true, 
    solver=SoftQ, # or SAC for continuous states 
    d_opt::NamedTuple=(), 
    log::NamedTuple=(;), 
    regularize::Bool=true,
    
    kwargs...)

    # Normalize and/or change device of expert and NDA data
    dev = device(π)
    A = action_space(π)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> dev

    # loss calculations
    V = soft_value(s)
    V' = soft_value(π,s')
    a_oh = Flux.one_hot(a)
    Q = value(π,s,a)
    y = 

    
    function SoftQ_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
        𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* soft_value(π, 𝒟[:sp])
    end



end