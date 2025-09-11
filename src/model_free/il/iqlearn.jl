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

function iq_loss(;γ=Float32(0.9),reg::Bool=true, α_reg=Float32(0.5), 
    gp::Bool=true, λ_gp=Float32(10))

    (π, 𝒫, 𝒟, y; info=Dict()) -> begin
        V = soft_value(π, 𝒟[:s])
        Vp = soft_value(π, 𝒟[:sp])
        Q = value(π, 𝒟[:s], 𝒟[:a])
        y = γ .* (1.f0 .- 𝒟[:done]) .* Vp
        R = Q-y
        expert = 𝒟[:expert][1,:]
        p1 = mean(-R[expert])
        p2 = mean(V - y)

        loss = p1+p2
        ignore_derivatives() do 
            info[:softQloss] = p1
            info[:valueloss] = p2
            info[:avg_R_expert_IQ] = -p1
            info[:avg_R_demo_IQ] = mean(R[.!expert])
        end

        if gp
            grad_pen = λ_gp*gradient_penalty(π.network, 𝒟[:s][:,expert], 𝒟[:s][:,.!expert])
            ignore_derivatives() do 
                info[:grad_pen] = grad_pen
            end
            loss += grad_pen
        end
        if reg
            reg_loss = 1/(4*α_reg) .* mean(R .^ 2)
            ignore_derivatives() do
                info[:reg_loss] = reg_loss
            end
            loss += reg_loss
        end
        loss
    end
end

# fixme - right way of giving buffers correct labels after each sampling 
# (only need to do expert once)
function iq_callback(𝒟; 𝒮, kwargs...)
    # add expert field to demo to be added to solver buffer
    𝒟[:expert] = Array(fill(false, 1, length(𝒟[:done])))
end

"""
Online Inverse Q-Learning solver.

```julia
OnlineIQLearn(;
    π, 
    S, 
    𝒟_demo, 
    γ=Float32(0.9),
    normalize_demo::Bool=true, 
    solver=SoftQ, # or SAC for continuous states 
    log::NamedTuple=(;period=500), 
    reg::Bool=true,
    α_reg=Float32(0.5),
    gp::Bool=true,
    λ_gp=Float32(10.),
    kwargs...)
```
"""
function OnlineIQLearn(;
        π, 
        S, 
        𝒟_demo, 
        γ=Float32(0.9),
        normalize_demo::Bool=true, 
        solver=SoftQ, # or SAC for continuous states 
        log::NamedTuple=(;period=500), 
        reg::Bool=true,
        α_reg=Float32(0.5),
        gp::Bool=true,
        λ_gp=Float32(10.),
        kwargs...)

    # Normalize and/or change device of expert and NDA data
    dev = device(π)
    A = action_space(π)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo.data[:expert] = Array(fill(true,1,length(𝒟_demo.data[:done])))
    𝒟_demo = 𝒟_demo |> dev

    solver(;π=π, 
        S=S, 
        extra_buffers=[𝒟_demo],
        buffer_fractions=[1/2, 1/2],
        log=(dir="log/iq", log...),
        c_loss=iq_loss(;γ=γ,reg=reg,α_reg=α_reg, gp=gp,λ_gp=λ_gp),
        target_fn=(args...;kwargs...)->nothing,
        post_sample_callback=iq_callback,
        required_columns = Symbol[:expert],
        kwargs...)
end