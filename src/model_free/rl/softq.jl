# exploration: action(s) propto softmax(q(s)/alpha) 

# update target = reward + (1-done)*gamma*soft_v(sp)
# soft_v(s) = alpha*logsumexp(q(s)/alpha)
# update q(s, a) to target

soft_value(π::DiscreteNetwork, s;α=Float32(1.)) = α .* logsumexp((value(π, s) ./ α), dims=1)

function softq_target(α)
    (π, 𝒫, 𝒟, γ::Float32; kwargs...) -> begin
        𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* soft_value(π, 𝒟[:sp];α=α)
    end
end

function SoftQ(;π::DiscreteNetwork, 
          N::Int, 
          ΔN=4, 
          c_opt::NamedTuple=(;epochs=4), 
          log::NamedTuple=(;),
          c_loss=td_loss(),
          α=Float32(1.),
          prefix="",
          kwargs...)

π.always_stochastic = true
π.logit_conversion = (π, s) -> softmax(value(π, s) ./ α)

OffPolicySolver(;agent=PolicyParams(π=π, π⁻=deepcopy(π)), 
                  log=LoggerParams(;dir="log/softq", log...),
                  N=N,
                  ΔN=ΔN,
                  c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), c_opt...),
                  target_fn=softq_target(α),
                  kwargs...)
end 
    




