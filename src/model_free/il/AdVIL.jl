function advil_π_loss(π, 𝒫, 𝒟; info=Dict())
    π_a = action(π, 𝒟[:s])
    mean(value(π, 𝒟[:s], π_a)) + 𝒫[:λ_BC]*Flux.mse(π_a, 𝒟[:a])
end

function advil_d_loss(π, 𝒫, 𝒟, y; info=Dict())
    π_sa = vcat(𝒟[:s], action(π, 𝒟[:s]))
    expert_sa = vcat(𝒟[:s], 𝒟[:a])
    mean(value(π, expert_sa)) - mean(value(π, π_sa)) + 𝒫[:λ_GP]*gradient_penalty(critic(π), expert_sa, π_sa, target=0.4f0)
end

"""
Adversarial Value Moment Imitation Learning (AdVIL) solver.

```julia
AdVIL(;
    π, 
    S,
    𝒟_demo, 
    normalize_demo::Bool=true, 
    λ_GP::Float32=10f0, 
    λ_orth::Float32=1f-4, 
    λ_BC::Float32=2f-1, 
    a_opt::NamedTuple=(;), 
    c_opt::NamedTuple=(;), 
    log::NamedTuple=(;), 
    kwargs...)
```
"""
function AdVIL(;
        π, 
        S,
        𝒟_demo, 
        normalize_demo::Bool=true, 
        λ_GP::Float32=10f0, 
        λ_orth::Float32=1f-4, 
        λ_BC::Float32=2f-1, 
        a_opt::NamedTuple=(;), 
        c_opt::NamedTuple=(;), 
        log::NamedTuple=(;), 
        kwargs...)
                
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)
    
    BatchSolver(;agent=PolicyParams(π),
                 S=S,
                 𝒫=(λ_GP=λ_GP, λ_BC=λ_BC,),
                 𝒟_train = 𝒟_demo,
                 a_opt=TrainingParams(;name="actor_", loss=advil_π_loss, regularizer=OrthogonalRegularizer(λ_orth), a_opt...),
                 c_opt=TrainingParams(;name="discriminator_", loss=advil_d_loss, c_opt...),
                 log=LoggerParams(;dir="log/AdVIL", period=1, log...),
                 kwargs...)
end

