@with_kw mutable struct DQNSolver <: Solver 
    π::DQNPolicy
    S::AbstractSpace
    A::AbstractSpace = action_space(π)
    N::Int = 1000
    rng::AbstractRNG = Random.GLOBAL_RNG
    exploration_policy::ExplorationPolicy = EpsGreedyPolicy(LinearDecaySchedule(start=1., stop=0.1, steps=N/2), rng, π.actions)
    L::Function = Flux.Losses.huber_loss
    regularizer = (θ) -> 0
    opt = ADAM(1e-3)
    batch_size::Int = 32
    max_steps::Int = 100
    eval_eps::Int = 10
    Δtrain::Int = 4 
    Δtarget_update::Int = 2000
    buffer_size = 1000
    buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size)
    buffer_init::Int = max(batch_size, 200)
    log::Union{Nothing, LoggerParams} = LoggerParams(dir = "log/dqn", period = 500)
    device = device(π)
    i::Int = 0
end

function POMDPs.solve(𝒮::DQNSolver, mdp, extra_buffers...)
    isprioritized = prioritized(𝒮.buffer)
    required_columns = isprioritized ? [:weight] : Symbol[]
    
    # Initialize minibatch buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.batch_size, required_columns, device = 𝒮.device)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, required_columns = required_columns, max_steps = 𝒮.max_steps, exploration_policy = 𝒮.exploration_policy, rng = 𝒮.rng)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i = 𝒮.i)
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.Δtrain, step = 𝒮.Δtrain)
        # Take Δtrain steps in the environment
        push!(𝒮.buffer, steps!(s, explore = true, i = 𝒮.i, Nsteps = 𝒮.Δtrain))
       
        # Sample a minibatch
        rand!(𝒮.rng, 𝒟, 𝒮.buffer, extra_buffers..., i = 𝒮.i)
        
        # Compute target, td_error and td_loss for backprop
        y = target(𝒮.π.Q⁻, 𝒟, γ)
        isprioritized && update_priorities!(𝒮.buffer, 𝒟.indices, cpu(td_error(𝒮.π, 𝒟, y)))
        info = train!(𝒮.π, (;kwargs...) -> td_loss(𝒮.π, 𝒟, y, 𝒮.L, isprioritized; kwargs...), 𝒮.opt, regularizer = 𝒮.regularizer)
        
        # Update target network
        elapsed(𝒮.i + 1:𝒮.i + 𝒮.Δtrain, 𝒮.Δtarget_update) && copyto!(𝒮.π.Q⁻, 𝒮.π.Q)
        
        # Log results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.Δtrain, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                            info,
                                            log_exploration(𝒮.exploration_policy, 𝒮.i))
    end
    𝒮.i += 𝒮.Δtrain
    𝒮.π
end

