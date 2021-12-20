@with_kw mutable struct OnPolicySolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 200 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    interaction_storage = nothing # If this is initialized to an array then it will store all interactions
    post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    post_batch_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling a batch
    
    # On-policy-specific parameters
    λ_gae::Float32 = 0.95 # Generalized advantage estimation parameter
    required_columns = Symbol[]# Extra data columns to store
    
    # Parameters specific to cost constraints (a separete value network)
    Vc::Union{ContinuousNetwork, Nothing} = nothing # Cost value approximator
    cost_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the cost value
end

function POMDPs.solve(𝒮::OnPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.agent.space, 𝒮.ΔN, 𝒮.required_columns, device=device(𝒮.agent.π))
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns, λ=λ, max_steps=𝒮.max_steps, Vc=𝒮.Vc)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, 𝒮=𝒮)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Info to collect during training
        info = Dict()
        
        # Sample transitions into the batch buffer
        steps!(s, 𝒟, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i, store=𝒮.interaction_storage, cb=(D) -> 𝒮.post_sample_callback(D, info=info, 𝒮=𝒮))
        
        # Post-batch callback, often used for additional training
        𝒮.post_batch_callback(𝒟, info=info, 𝒮=𝒮)
        
        # Train parameters
        for (θs, p_opt) in 𝒮.param_optimizers
            batch_train!(θs, p_opt, 𝒮.𝒫, 𝒟, info=info, π_loss=𝒮.agent.π)
        end
        
        # Train the actor
        batch_train!(actor(𝒮.agent.π), 𝒮.a_opt, 𝒮.𝒫, 𝒟, info=info)
        
        # Train the critic (if applicable)
        if !isnothing(𝒮.c_opt)
            batch_train!(critic(𝒮.agent.π), 𝒮.c_opt, 𝒮.𝒫, 𝒟, info=info)
        end
        
        if !isnothing(𝒮.cost_opt)
            batch_train!(𝒮.Vc, 𝒮.cost_opt, 𝒮.𝒫, 𝒟, info=info)
        end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, 𝒮=𝒮)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.agent.π
end

