@with_kw mutable struct AdVILSolver <: Solver
    π # The policy to train
    D # Discriminator
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    max_steps::Int = 100 # Maximum number of steps per episode
    𝒟_expert # training data
    a_opt::TrainingParams # Training parameters for the actor
    d_opt::TrainingParams # Training parameters for the discriminator
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i = 0 # Number of epochs of training
end

function AdVIL_π_loss(λ_BC::Float32)
    (π, D, 𝒟; info=Dict())->begin 
        π_a = action(π, 𝒟[:s])
        mean(value(D, 𝒟[:s], π_a)) + λ_BC*Flux.mse(π_a, 𝒟[:a])
    end
end

function AdVIL_D_loss(λ_GP::Float32)
    (π, D, 𝒟; info=Dict()) -> begin
        π_sa = vcat(𝒟[:s], action(π, 𝒟[:s]))
        expert_sa = vcat(𝒟[:s], 𝒟[:a])
        mean(value(D, expert_sa)) - mean(value(D, π_sa)) + λ_GP*gradient_penalty(D, expert_sa, π_sa, target=0.4f0)
    end
end

function AdVIL(;λ_GP::Float32=10f0, λ_orth::Float32=1f-4, λ_BC::Float32=2f-1, a_opt::NamedTuple=(;), d_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    AdVILSolver(;a_opt=TrainingParams(;name="actor_", loss=AdVIL_π_loss(λ_BC), regularizer=OrthogonalRegularizer(λ_orth), a_opt...),
                 d_opt=TrainingParams(;name="discriminator_", loss=AdVIL_D_loss(λ_GP), d_opt...),
                 log=LoggerParams(;dir="log/AdVIL", period=1, log...),
                 kwargs...)
end

function POMDPs.solve(𝒮::AdVILSolver, mdp)
    # Minibatch buffer
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.a_opt.batch_size, device=device(𝒮.π))
    
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps)
    
    # Loop over the number of epochs
    infos = []
    for 𝒮.i=𝒮.i:𝒮.i + 𝒮.a_opt.epochs
        rand!(𝒟, 𝒮.𝒟_expert) # fill minibatch buffer
        info_a = train!(𝒮.π, (;kwargs...)->𝒮.a_opt.loss(𝒮.π, 𝒮.D, 𝒟; kwargs...), 𝒮.a_opt) 
        info_d = train!(𝒮.D, (;kwargs...)->𝒮.d_opt.loss(𝒮.π, 𝒮.D, 𝒟; kwargs...), 𝒮.d_opt)
        push!(infos, merge(info_a, info_d))
        log(𝒮.log, 𝒮.i, infos[end], s=s) # Log the results
    end
    
    𝒮.π
end





