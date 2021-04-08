function AdVIL_π_loss(λ_BC::Float32)
    (π, 𝒟; info=Dict())->begin 
        π_a = action(π, 𝒟[:s])
        mean(value(π, 𝒟[:s], π_a)) + λ_BC*Flux.mse(π_a, 𝒟[:a])
    end
end

function AdVIL_D_loss(λ_GP::Float32)
    (π, 𝒟; info=Dict()) -> begin
        π_sa = vcat(𝒟[:s], action(π, 𝒟[:s]))
        expert_sa = vcat(𝒟[:s], 𝒟[:a])
        mean(value(π, expert_sa)) - mean(value(π, π_sa)) + λ_GP*gradient_penalty(critic(π), expert_sa, π_sa, target=0.4f0)
    end
end

function AdVIL(;π, S, A=action_space(π), 𝒟_expert, λ_GP::Float32=10f0, λ_orth::Float32=1f-4, λ_BC::Float32=2f-1, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    𝒟_expert = normalize!(deepcopy(𝒟_expert), S, A) |> device(π)
    BatchSolver(;π=π,
                 S=S,
                 A=A,
                 𝒟_train = 𝒟_expert,
                 a_opt=TrainingParams(;name="actor_", loss=AdVIL_π_loss(λ_BC), regularizer=OrthogonalRegularizer(λ_orth), a_opt...),
                 c_opt=TrainingParams(;name="discriminator_", loss=AdVIL_D_loss(λ_GP), c_opt...),
                 log=LoggerParams(;dir="log/AdVIL", period=1, log...),
                 kwargs...)
end

