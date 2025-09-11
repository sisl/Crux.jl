"""
Batch solver type.

Fields
======
- `agent::PolicyParams` Policy parameters ([`PolicyParams`](@ref))
- `S::AbstractSpace` State space
- `max_steps::Int = 100` Maximum number of steps per episode
- `𝒟_train` Training data
- `param_optimizers::Dict{Any, TrainingParams} = Dict()` Training parameters for the parameters
- `a_opt::TrainingParams` Training parameters for the actor
- `c_opt::Union{Nothing, TrainingParams} = nothing` Training parameters for the discriminator
- `target_fn = nothing` the target function for value-based methods
- `target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0)` Function for updating the target network
- `𝒫::NamedTuple = (;)` Parameters of the algorithm
- `log::Union{Nothing, LoggerParams} = nothing` The logging parameters
- `required_columns = Symbol[]` Extra columns to sample
- `epoch = 0` Number of epochs of training
"""
@with_kw mutable struct BatchSolver <: Solver
    agent::PolicyParams
    S::AbstractSpace # State space
    max_steps::Int = 100 # Maximum number of steps per episode
    𝒟_train # training data
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the discriminator
    target_fn = nothing # the target function for value-based methods
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    required_columns = Symbol[] # Extra columns to sample
    epoch = 0 # Number of epochs of training
end

function POMDPs.solve(𝒮::BatchSolver, mdp)
    γ = Float32(discount(mdp))
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, max_steps=𝒮.max_steps, required_columns=𝒮.required_columns)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)
    
    # Log initial performance
    log(𝒮.log, 𝒮.epoch, 𝒮=𝒮)
    
    # Loop over the number of epochs
    infos = []
    grad_steps = 0
    for 𝒮.epoch=𝒮.epoch:𝒮.epoch + 𝒮.a_opt.epochs
        minibatch_infos = [] # stores the info from each minibatch
        
        # Shuffle the experience buffer
        shuffle!(𝒮.𝒟_train)
        
        # Call train for each minibatch
        batches = partition(1:length(𝒮.𝒟_train), 𝒮.a_opt.batch_size)
        for batch in batches
            mb = minibatch(𝒮.𝒟_train, batch)
            info = Dict()
            
            # Train parameters
            for (θs, p_opt) in 𝒮.param_optimizers
                train!(θs, (;kwargs...) -> p_opt.loss(𝒮.agent.π, 𝒮.𝒫, mb; kwargs...), p_opt, info=info)
            end
            
            # Compute target
            y = !isnothing(𝒮.target_fn) ? 𝒮.target_fn(𝒮.agent.π⁻, 𝒮.𝒫, mb, γ) : nothing
            
            # Optionally train the critic
            if !isnothing(𝒮.c_opt)
                train!(critic(𝒮.agent.π), (;kwargs...)->𝒮.c_opt.loss(𝒮.agent.π, 𝒮.𝒫, mb, y; kwargs...), 𝒮.c_opt, info=info)
                
                if !isnothing(y)
                    𝒮.target_update(𝒮.agent.π⁻, 𝒮.agent.π)
                end
            end 
            
            # Train the actor
            train!(actor(𝒮.agent.π), (;kwargs...)->𝒮.a_opt.loss(𝒮.agent.π, 𝒮.𝒫, mb; kwargs...), 𝒮.a_opt, info=info)
            
            grad_steps += 1
            log(𝒮.log, grad_steps, info, 𝒮=𝒮)
            
            push!(minibatch_infos, info)
        end
        push!(infos, aggregate_info(minibatch_infos))                
        
        # Early stopping
        𝒮.a_opt.early_stopping(infos) && break
    end
    
    𝒮.agent.π
end



