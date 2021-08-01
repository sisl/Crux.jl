@with_kw mutable struct BatchSolver <: Solver
    π # The policy to train
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    max_steps::Int = 100 # Maximum number of steps per episode
    𝒟_train # training data
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the discriminator
    target_fn = nothing # the target function for value-based methods
    π⁻ = deepcopy(π) # use a target policy for value-bsaed methods
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    required_columns = Symbol[] # Extra columns to sample
    epoch = 0 # Number of epochs of training
    
end

function POMDPs.solve(𝒮::BatchSolver, mdp)
    γ = Float32(discount(mdp))
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, required_columns=𝒮.required_columns)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)
    
    # Log initial performance
    log(𝒮.log, 𝒮.epoch)
    
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
                train!(θs, (;kwargs...) -> p_opt.loss(𝒮.π, 𝒮.𝒫, mb; kwargs...), p_opt, info=info)
            end
            
            # Compute target
            y = !isnothing(𝒮.target_fn) ? 𝒮.target_fn(𝒮.π⁻, 𝒮.𝒫, mb, γ) : nothing
            
            # Optionally train the critic
            if !isnothing(𝒮.c_opt)
                train!(critic(𝒮.π), (;kwargs...)->𝒮.c_opt.loss(𝒮.π, 𝒮.𝒫, mb, y; kwargs...), 𝒮.c_opt, info=info)
                
                if !isnothing(y)
                    𝒮.target_update(𝒮.π⁻, 𝒮.π)
                end
            end 
            
            # Train the actor
            train!(actor(𝒮.π), (;kwargs...)->𝒮.a_opt.loss(𝒮.π, 𝒮.𝒫, mb; kwargs...), 𝒮.a_opt, info=info)
            
            grad_steps += 1
            log(𝒮.log, grad_steps, info)
            
            push!(minibatch_infos, info)
        end
        push!(infos, aggregate_info(minibatch_infos))                
        
        # Early stopping
        𝒮.a_opt.early_stopping(infos) && break
    end
    
    𝒮.π
end

# Early stopping function that terminates training on validation error increase
function stop_on_validation_increase(π, 𝒫, 𝒟_val, loss; window=5)
    k = "validation_error"
    (infos) -> begin
        ve = loss(π, 𝒫, 𝒟_val) # Compute the validation error
        infos[end][k] = ve # store it
        N = length(infos)
        if length(infos) >= 2*window
            curr_window = mean([infos[i][k] for i=N-window+1:N])
            old_window = mean([infos[i][k] for i=N-2*window+1:N-window])
            return curr_window >= old_window # check if the error has gone up
        end
        false
    end
end

