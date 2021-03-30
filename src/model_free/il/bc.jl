@with_kw mutable struct BCSolver <: Solver
    π # The policy to train
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    max_steps::Int = 100 # Maximum number of steps per episode
    𝒟_expert # training data
    opt::TrainingParams # Training parameters
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i = 0 # Number of epochs of training
end

mse_action_loss() = (π, 𝒟; kwargs...) -> Flux.mse(action(π, 𝒟[:s]), 𝒟[:a])
function mse_value_loss(λe::Float32)
    (π, 𝒟; kwargs...) -> begin
        eloss = -mean(entropy(π, 𝒟[:s]))
        mseloss = Flux.mse(value(π, 𝒟[:s]), 𝒟[:value])
        λe*eloss + mseloss
    end
end
function logpdf_bc_loss(λe::Float32)
    (π, 𝒟; kwargs...)->begin
        eloss = -mean(entropy(π, 𝒟[:s]))
        lloss = -mean(logpdf(π, 𝒟[:s], 𝒟[:a]))
        λe*eloss + lloss
    end
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

function BC(;π, 𝒟_expert, loss=nothing, validation_fraction=0.3, window=100, λe::Float32=1f-3, opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    if isnothing(loss)
        loss = π isa ContinuousNetwork ? mse_action_loss() : logpdf_bc_loss(λe)
    end
    shuffle!(𝒟_expert)
    𝒟_train, 𝒟_validate = split(𝒟_expert, [1-validation_fraction, validation_fraction])
    BCSolver(;π=π, 
              𝒟_expert=𝒟_train, 
              opt=TrainingParams(;early_stopping=stop_on_validation_increase(π, 𝒟_validate, loss, window=window), loss=loss, opt...), 
              log=LoggerParams(;dir="log/bc", period=1, log...),
              kwargs...)
end

function POMDPs.solve(𝒮::BCSolver, mdp)
    # Minibatch buffer
    𝒟 = buffer_like(𝒮.𝒟_expert, capacity=𝒮.opt.batch_size, device=device(𝒮.π))
    
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.π, max_steps=𝒮.max_steps)
    
    # Loop over the number of epochs
    infos = []
    for 𝒮.i=𝒮.i:𝒮.i + 𝒮.opt.epochs
        rand!(𝒟, 𝒮.𝒟_expert) # fill minibatch buffer
        push!(infos, train!(𝒮.π, (;kwargs...)->𝒮.opt.loss(𝒮.π, 𝒟; kwargs...), 𝒮.opt)) # take training step
        𝒮.opt.early_stopping(infos) && break # Stop early
        log(𝒮.log, 𝒮.i, infos[end], s=s) # Log the results
    end
    
    𝒮.π
end

