@with_kw mutable struct BatchSolver <: Solver
    π # The policy to train
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    max_steps::Int = 100 # Maximum number of steps per episode
    𝒟_train # training data
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the discriminator
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    required_columns = Symbol[] # Extra columns to sample
    epoch = 0 # Number of epochs of training
end

function POMDPs.solve(𝒮::BatchSolver, mdp)    
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, required_columns=𝒮.required_columns)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)
    
    # Log initial performance
    log(𝒮.log, 𝒮.epoch)
    
    # Loop over the number of epochs
    infos = []
    for 𝒮.epoch=𝒮.epoch:𝒮.epoch + 𝒮.a_opt.epochs
        minibatch_infos = [] # stores the info from each minibatch
        
        # Shuffle the experience buffer
        shuffle!(𝒮.𝒟_train)
        
        # Call train for each minibatch
        batches = partition(1:length(𝒮.𝒟_train), 𝒮.a_opt.batch_size)
        for batch in batches
            mb = minibatch(𝒮.𝒟_train, batch)
            # Train the actor
            info = train!(actor(𝒮.π), (;kwargs...)->𝒮.a_opt.loss(𝒮.π, mb; kwargs...), 𝒮.a_opt)
            
            # Optionally train the critic
            if !isnothing(𝒮.c_opt)
                info_d = train!(critic(𝒮.π), (;kwargs...)->𝒮.c_opt.loss(𝒮.π, mb; kwargs...), 𝒮.c_opt)
                info = merge(info, info_d)
            end 
            push!(minibatch_infos, info)
        end
        push!(infos, aggregate_info(minibatch_infos))                
        
        # Early stopping
        𝒮.a_opt.early_stopping(infos) && break
        
        # Log the results
        log(𝒮.log, 𝒮.epoch+1, infos[end])
    end
    
    𝒮.π
end

# Early stopping function that terminates training on validation error increase
function stop_on_validation_increase(π, 𝒟_val, loss; window=5)
    k = "validation_error"
    (infos) -> begin
        ve = loss(π, 𝒟_val) # Compute the validation error
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
