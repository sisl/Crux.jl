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

mse_bc_loss(π, 𝒟; kwargs...) = Flux.mse(action(π, 𝒟[:s]), 𝒟[:a])
function logpdf_bc_loss(λe::Float32)
    (π, 𝒟; kwargs...)->begin
        eloss = -mean(entropy(π, 𝒟[:s]))
        lloss = -mean(logpdf(π, 𝒟[:s], 𝒟[:a]))
        λe*eloss + lloss
    end
end

function BC(;π, 𝒟_expert, loss=nothing, validation_fraction=0.3, λe::Float32=1f-3, opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    if isnothing(loss)
        loss = π isa ContinuousNetwork ? mse_bc_loss : logpdf_bc_loss(λe)
    end
    shuffle!(𝒟_expert)
    𝒟_train, 𝒟_validate = split(𝒟_expert, [1-validation_fraction, validation_fraction])    
    BCSolver(;π=π, 
              𝒟_expert=𝒟_train, 
              opt=TrainingParams(;early_stopping=stop_on_validation_increase(π, 𝒟_validate, loss), loss=loss, opt...), 
              log=LoggerParams(;dir="log/bc", period=1, log...),
              kwargs...)
end

function POMDPs.solve(𝒮::BCSolver, mdp)
    # Minibatch buffer
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.opt.batch_size, device=device(𝒮.π))
    
    # Sampler for logging performance
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps=𝒮.max_steps)
    
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

