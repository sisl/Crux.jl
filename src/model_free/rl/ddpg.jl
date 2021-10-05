# Set yᵢ = rᵢ + γQ′(sᵢ₊₁, μ′(sᵢ₊₁ | θᵘ′) | θᶜ′)
function DDPG_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(π, 𝒟[:sp], action(π, 𝒟[:sp]))
end

function smoothed_DDPG_target(π, 𝒫, 𝒟, γ::Float32; i)
    ap, _ = exploration(𝒫[:π_smooth], 𝒟[:sp], π_on=π, i=i)
    y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(π, 𝒟[:sp], ap)
end

# ∇_θᵘ 𝐽 ≈ 1/𝑁 Σᵢ ∇ₐQ(s, a | θᶜ)|ₛ₌ₛᵢ, ₐ₌ᵤ₍ₛᵢ₎ ∇_θᵘ μ(s | θᵘ)|ₛᵢ
DDPG_actor_loss(π, 𝒫, 𝒟; info=Dict()) = -mean(value(π, 𝒟[:s], action(π, 𝒟[:s])))

# T. P. Lillicrap, et al., "Continuous control with deep reinforcement learning", ICLR 2016.
function DDPG(;π::ActorCritic, 
               ΔN=50, 
               π_explore=GaussianNoiseExplorationPolicy(0.1f0),  
               a_opt::NamedTuple=(;), 
               c_opt::NamedTuple=(;),
               a_loss=DDPG_actor_loss,
               c_loss=td_loss(),
               target_fn=DDPG_target,
               prefix="",
               log::NamedTuple=(;), 
               π_smooth=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), kwargs...)
               
    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
                     ΔN=ΔN,
                     𝒫=(π_smooth=π_smooth,),
                     log=LoggerParams(;dir = "log/ddpg", log...),
                     a_opt=TrainingParams(;loss=a_loss, name=string(prefix, "actor_"), a_opt...),
                     c_opt=TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                     target_fn=target_fn,
                     kwargs...)
end 
        

