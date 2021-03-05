@with_kw mutable struct ValueDICESolver <: Solver
    π # Policy
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 4 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = LoggerParams(;dir = "log/valueDICE") # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::TrainingParams # Training parameters for the critic
    
    𝒟_expert # expert buffer
    α::Float32 = 0.1 # mixing parameter
    π_explore=π
    buffer_size = 1000 # Size of the buffer
    buffer::ExperienceBuffer=ExperienceBuffer(S, A, buffer_size,[:t]) # The replay buffer
    buffer_init::Int=max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
end

ValueDICE(;ΔN=50, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), kwargs...) = ValueDICESolver(;ΔN=ΔN, a_opt=TrainingParams(;loss=valueDICE_loss, a_opt...), c_opt=TrainingParams(;loss=valueDICE_loss, epochs=ΔN, c_opt...), kwargs...)

# orthogonal initialization
# GP on critic
# orthogonal regularization on the policy

function valueDICE_loss(π, 𝒟, 𝒟_exp, α, γ; info=Dict())
    a0, _= exploration(π.A, 𝒟_exp[:s]) #:s0
    a, _ = exploration(π.A, 𝒟[:sp])
    ae, _  = exploration(π.A, 𝒟_exp[:sp])
    
    ΔνE = value(π, 𝒟_exp[:s], 𝒟_exp[:a]) - γ*value(π, 𝒟_exp[:sp], ae)
    Δν = value(π, 𝒟[:s], 𝒟[:a]) - γ*value(π, 𝒟[:sp], a)
    
    
    Jlog = log(mean((1-α)*exp.(ΔνE) .+ α*exp.(Δν)))
    Jlin = mean((1-α)*(1-γ)*value(π, 𝒟_exp[:s], a0) + α.*Δν)
    
    Jlog - Jlin
end

function POMDPs.solve(𝒮::ValueDICESolver, mdp, logmdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.c_opt.batch_size, [:t], device=device(𝒮.π))
    𝒟_exp = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.c_opt.batch_size, [:t, :s0], device=device(𝒮.π))
    𝒟_exp.data[:expert_val] = ones(Float32, 1, 𝒮.c_opt.batch_size)
    
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore, required_columns=[:t])
    slog = Sampler(logmdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore, required_columns=[:t])

    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, s=slog)

    # Fill the buffer with initial observations before training
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
    
    # Loop over the desired number of environment interactions
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions into the replay buffer
        push!(𝒮.buffer, steps!(s, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i))

        infos = []
        # Loop over the desired number of training steps
        for epoch in 1:𝒮.c_opt.epochs
            # geometric_sample!(𝒟, 𝒮.buffer, γ)
            # geometric_sample!(𝒟_exp, 𝒮.𝒟_expert, γ)
            # 
            rand!(𝒟, 𝒮.buffer)
            rand!(𝒟_exp, 𝒮.𝒟_expert)
            
            # Update the critic and actor
            info_c = train!(𝒮.π.C, (;kwargs...) -> 𝒮.c_opt.loss(𝒮.π, 𝒟, 𝒟_exp, 𝒮.α, γ; kwargs...), 𝒮.c_opt)
            info_a = train!(𝒮.π.A, (;kwargs...) -> -𝒮.a_opt.loss(𝒮.π, 𝒟, 𝒟_exp, 𝒮.α, γ; kwargs...), 𝒮.a_opt)
            
            push!(infos, merge(info_c, info_a))            
        end
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, aggregate_info(infos), s=slog)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

