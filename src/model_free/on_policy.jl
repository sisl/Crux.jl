@with_kw mutable struct OnPolicySolver <: Solver
    π # Policy
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 200 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the critic
    
    # On-policy-specific parameters
    λ_gae::Float32 = 0.95 # Generalized advantage estimation parameter
    required_columns = isnothing(c_opt) ? [:return, :logprob] : [:return, :advantage, :logprob] # Extra data columns to store
    post_batch_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling a batch
end

function POMDPs.solve(𝒮::OnPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device=device(𝒮.π))
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.π, required_columns=𝒮.required_columns, λ=λ, max_steps=𝒮.max_steps, π_explore=𝒮.π)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, s=s)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions into the batch buffer
        push!(𝒟, steps!(s, Nsteps=𝒮.ΔN, reset=true, explore=true, i=𝒮.i))
        
        # Call the post-batch callback function
        info_cb = Dict()
        𝒮.post_batch_callback(𝒟, info=info_cb)
        
        # Train the actor
        info = batch_train!(𝒮.π, 𝒮.a_opt, 𝒟)
        
        # Train the critic (if applicable)
        if !isnothing(𝒮.c_opt)
            info_c = batch_train!(𝒮.π, 𝒮.c_opt, 𝒟)
            merge!(info, info_c)
        end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info, info_cb, s=s)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

