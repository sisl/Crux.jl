"""
DQN target function.
"""
function dqn_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
      𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(value(π, 𝒟[:sp]), dims=1)
end


"""
Deep Q-learning (DQN) solver.
- V. Mnih, et al., "Human-level control through deep reinforcement learning", Nature 2015.

```julia
DQN(;
      π::DiscreteNetwork, 
      N::Int, 
      ΔN=4, 
      π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs), 
      c_opt::NamedTuple=(;), 
      log::NamedTuple=(;),
      c_loss=td_loss(),
      target_fn=dqn_target,
      prefix="",
      kwargs...)
```
"""
function DQN(;
            π::DiscreteNetwork, 
            N::Int, 
            ΔN=4, 
            π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs), 
            c_opt::NamedTuple=(;), 
            log::NamedTuple=(;),
            c_loss=td_loss(),
            target_fn=dqn_target,
            prefix="",
            kwargs...)
              
      OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
                      log=LoggerParams(;dir="log/dqn", log...),
                      N=N,
                      ΔN=ΔN,
                      c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                      target_fn=target_fn,
                      kwargs...)
end 
        

