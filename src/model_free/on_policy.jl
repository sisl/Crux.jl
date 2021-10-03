@with_kw mutable struct OnPolicySolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 200 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    
    # On-policy-specific parameters
    λ_gae::Float32 = 0.95 # Generalized advantage estimation parameter
    required_columns = isnothing(c_opt) ? [:return, :logprob] : [:return, :advantage, :logprob] # Extra data columns to store
    post_batch_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling a batch
    loop_start_callback = (𝒮) -> nothing # Callback that happens at the beginning of each experience gathering iteration
end

function POMDPs.solve(𝒮::OnPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device=device(𝒮.agent.π))
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns, λ=λ, max_steps=𝒮.max_steps)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Call the loop start callback function
        𝒮.loop_start_callback(𝒮)
        
        # Sample transitions into the batch buffer
        push!(𝒟, steps!(s, Nsteps=𝒮.ΔN, reset=true, explore=true, i=𝒮.i))
        
        # Info to collect during training
        info = Dict()
        
        # Call the post-batch callback function
        𝒮.post_batch_callback(𝒟, info=info)
        
        # Train the actor
        batch_train!(actor(𝒮.agent.π), 𝒮.a_opt, 𝒮.𝒫, 𝒟, info=info)
        
        # Train the critic (if applicable)
        if !isnothing(𝒮.c_opt)
            batch_train!(critic(𝒮.agent.π), 𝒮.c_opt, 𝒮.𝒫, 𝒟, info=info)
        end
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.agent.π
end

