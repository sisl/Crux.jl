@with_kw mutable struct DQNSolver <: Solver 
    π::DQNPolicy
    s_dim::Int
    a_dim::Int
    N::Int = 1000
    exploration_policy::ExplorationPolicy
    device = cpu
    rng::AbstractRNG = Random.GLOBAL_RNG
    L::Function = Flux.Losses.huber_loss
    opt = ADAM(1e-3)
    batch_size::Int = 32
    Δtrain::Int = 4 
    Δtarget_update::Int = 2000
    buffer_init::Int = max(batch_size, 200)
    log = LoggerParams(dir = "log/dqn", period = 500, rng = rng)
    buffer::ExperienceBuffer = ExperienceBuffer(mdp, 1000, device = device)
    i::Int = 1
end

target(Q⁻, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q⁻(𝒟[:sp]), dims=1)

q_predicted(π, 𝒟) = sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1)

td_loss(π, 𝒟, y, L) = L(q_predicted(π, 𝒟), y)

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)

#TODO: Look at RL class DQN pong for inspo on gpu usage and frame processing
function POMDPs.solve(𝒮::DQNSolver, mdp)
    # Log the pre-train performance
    𝒮.i == 1 && log(𝒮.log, 0, mdp, 𝒮.π)
    
    # Initialize minibatch buffer and sampler
    𝒟 = ExperienceBuffer(mdp, 𝒮.batch_size, device = 𝒮.device, indices = prioritized(𝒮.buffer))
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.max_steps, exploration_policy = 𝒮.exploration_policy, rng = 𝒮.rng)
    
    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i = 𝒮.i)
    
    for 𝒮.i = range(𝒮.i, length = 𝒮.N, step = 𝒮.Δtrain)
        # Take a step in the environment
        push!(𝒮.buffer, steps!(s, i = 𝒮.i, Nsteps = 𝒮.Δtrain))
        
        # Sample a minibatch
        rand!(𝒮.rng, 𝒟, buffer, i = 𝒮.i)
        
        # Compute target, td_error and td_loss for backprop
        y = target(𝒮.Q⁻, 𝒟, γ)
        prioritized(𝒮.buffer) && update_priorities!(buffer, 𝒟.indices, td_error(𝒮.π, 𝒟, y))
        loss, grad = train!(𝒮.π, () -> td_loss(𝒮.π, 𝒟, y, 𝒮.L), 𝒮.opt, 𝒮.device)
        
        # Update target network 
        elapsed(𝒮.i - 𝒮.Δtrain + 1:𝒮.i, 𝒮.Δtarget_update) && copyto!(𝒮.Q⁻, 𝒮.π.Q)
        
        # Log results
        log(𝒮.log, 𝒮.i, mdp, 𝒮.π, data = [logloss(loss, grad), logexploration(𝒮.exploration_policy, 𝒮.i)])
    end
    𝒮.π
end

