@with_kw mutable struct AdversarialOffPolicySolver <: Solver
    𝒮_pro::OffPolicySolver # Solver parameters for the protagonist
    𝒮_ant::OffPolicySolver # Solver parameters for the antagonist
    px::PolicyParams # Nominal disturbance policy
    train_pro_every::Int = 1
    train_ant_every::Int = 1
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
end

function POMDPs.solve(𝒮::AdversarialOffPolicySolver, mdp)
    # make some assertions on the parameters that should be shared
    @assert 𝒮.𝒮_pro.buffer == 𝒮.𝒮_ant.buffer
    @assert 𝒮.𝒮_pro.buffer_init == 𝒮.𝒮_ant.buffer_init
    @assert 𝒮.𝒮_pro.required_columns == 𝒮.𝒮_ant.required_columns
    @assert 𝒮.𝒮_pro.N == 𝒮.𝒮_ant.N
    @assert 𝒮.𝒮_pro.S == 𝒮.𝒮_ant.S
    @assert 𝒮.𝒮_pro.max_steps == 𝒮.𝒮_ant.max_steps
    
    # define shared parameters for easy reference
    buffer = 𝒮.𝒮_pro.buffer
    buffer_init = 𝒮.𝒮_pro.buffer_init
    ΔN = 𝒮.𝒮_pro.ΔN + 𝒮.𝒮_ant.ΔN
    N = 𝒮.𝒮_pro.N
    S = 𝒮.𝒮_pro.S
    max_steps = 𝒮.𝒮_pro.max_steps
    
    # Construct the training buffern for the protagonist and antagonist
    𝒟_pro = buffer_like(buffer, capacity=𝒮.𝒮_protagonist.c_opt.batch_size, device=device(𝒮.𝒮_protagonist.π))
    𝒟_ant = buffer_like(buffer, capacity=𝒮.𝒮_protagonist.c_opt.batch_size, device=device(𝒮.𝒮_protagonist.π))
    
    # Get the discount factor 
    γ = Float32(discount(mdp))
    
    # Two samplers, one for the traditional ap
    s_pro = Sampler(mdp, 𝒮.𝒮_pro.agent, adversary=𝒮.px, S=S, max_steps=max_steps, required_columns=extra_columns(buffer))
    s_ant = Sampler(mdp, 𝒮.𝒮_pro.agent, adversary=𝒮.𝒮_ant.agent, S=S, max_steps=max_steps, required_columns=extra_columns(buffer))
    
    s_nom = Sampler(mdp_nom, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, required_columns=setdiff(extra_columns(𝒮.buffer), [:x]))
    push!(𝒮.log.fns, (;kwargs...) -> Dict("nominal_undiscounted_return" => undiscounted_return(s_nom, Neps=10)))
    
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s_pro)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i)

    # Fill the buffer with initial observations before training
    𝒮.i += fillto!(buffer, s_pro, buffer_init, i=𝒮.i, explore=true)
    𝒮.𝒮_pro.i = 𝒮.i
    𝒮.𝒮_ant.i = 𝒮.i
    
    # Loop over the desired number of environment interactions
    for 𝒮.i in range(𝒮.i, stop=𝒮.i + N - ΔN, step=ΔN)
        # Update the per solver iteration
        𝒮.𝒮_pro.i = 𝒮.i
        𝒮.𝒮_ant.i = 𝒮.i
        
        # Sample transitions into the replay buffer
        push!(buffer, steps!(s_pro, Nsteps=𝒮.𝒮_pro.ΔN, explore=true, i=𝒮.i))
        push!(buffer, steps!(s_ant, Nsteps=𝒮.𝒮_ant.ΔN, explore=true, i=𝒮.i))
        
        # callback for potentially updating the buffer
        𝒮.post_experience_callback(buffer) 
        
        # Train the networks
        infos = []
        elapsed(𝒮.i + 1:𝒮.i + ΔN, 𝒮.train_pro_every) && merge!(infos, train_step(𝒮.𝒮_pro, 𝒟_pro, γ))
        elapsed(𝒮.i + 1:𝒮.i + ΔN, 𝒮.train_ant_every) && merge!(infos, train_step(𝒮.𝒮_ant, 𝒟_ant, γ))
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + ΔN, aggregate_info(infos))
    end
    𝒮.i += 𝒮.ΔN
    𝒮.𝒮_pro.i = 𝒮.i
    𝒮.𝒮_ant.i = 𝒮.i
    
    𝒮.𝒮_pro.π
end