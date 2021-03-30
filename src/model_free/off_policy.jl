@with_kw mutable struct OffPolicySolver <: Solver
    π # Policy
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 4 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
    c_opt::TrainingParams # Training parameters for the critic
    
    # Off-policy-specific parameters
    π⁻ = deepcopy(π)
    π_explore::Policy # exploration noise
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    buffer_size = 1000 # Size of the buffer
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size) # The replay buffer
    buffer_init::Int = max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
    extra_buffers = [] # extra buffers (i.e. for experience replay in continual learning)
    buffer_fractions = [1.0] # Fraction of the minibatch devoted to each buffer
end

function POMDPs.solve(𝒮::OffPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.c_opt.batch_size, device=device(𝒮.π))
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, s=s)

    # Fill the buffer with initial observations before training
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
    
    # Loop over the desired number of environment interactions
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions into the replay buffer
        push!(𝒮.buffer, steps!(s, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i))
        infos = []
        # Loop over the desired number of training steps
        for epoch in 1:𝒮.c_opt.epochs
            # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
            rand!(𝒟, 𝒮.buffer, 𝒮.extra_buffers..., fracs=𝒮.buffer_fractions, i=𝒮.i)
            
            # Compute target
            y = 𝒮.target_fn(𝒮.π⁻, 𝒟, γ, i=𝒮.i)
            
            # Update priorities (for prioritized replay)
            (ispri = isprioritized(𝒮.buffer)) && update_priorities!(𝒮.buffer, 𝒟.indices, cpu(td_error(𝒮.π, 𝒟, y)))
            
            # Train the critic
            info = train!(𝒮.π, (;kwargs...) -> 𝒮.c_opt.loss(𝒮.π, 𝒟, y; weighted=ispri, kwargs...), 𝒮.c_opt)
            
            # Train the actor 
            if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
                info_a = train!(𝒮.π.A, (;kwargs...) -> 𝒮.a_opt.loss(𝒮.π, 𝒟; kwargs...), 𝒮.a_opt)
                info = merge(info, info_a)
            
                # Update the target network
                𝒮.target_update(𝒮.π⁻, 𝒮.π)
            end
            
            # Store the training information
            push!(infos, info)
            
        end
        # If not using a separate actor, update target networks after critic training
        isnothing(𝒮.a_opt) && 𝒮.target_update(𝒮.π⁻, 𝒮.π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, aggregate_info(infos), s=s)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

function td_loss(π, 𝒟, y; loss=Flux.mse, weighted=false, name=:Qavg, info=Dict())
    Q = value(π, 𝒟[:s], 𝒟[:a]) 
    
    # Store useful information
    ignore() do
        info[name] = mean(Q)
    end
    
    loss(Q, y, agg = weighted ? weighted_mean(𝒟[:weight]) : mean)
end

function double_Q_loss(π, 𝒟, y; info=Dict(), weighted=false)
    q1loss = td_loss(π.C.N1, 𝒟, y, info=info, name=:Q1avg, weighted=weighted)
    q2loss = td_loss(π.C.N2, 𝒟, y, info=info, name=:Q2avg, weighted=weighted)
    q1loss + q2loss
end

td_error(π, 𝒟, y) = abs.(value(π, 𝒟[:s], 𝒟[:a])  .- y)


