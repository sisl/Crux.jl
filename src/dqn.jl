@with_kw mutable struct DQNSolver <: Solver 
    π::DQNPolicy
    sdim::Int
    adim::Int = length(π.actions)
    N::Int = 1000
    rng::AbstractRNG = Random.GLOBAL_RNG
    exploration_policy::ExplorationPolicy = EpsGreedyPolicy(LinearDecaySchedule(start=1., stop=0.1, steps=N/2), rng, π.actions)
    L::Function = Flux.Losses.huber_loss
    opt = ADAM(1e-3)
    batch_size::Int = 32
    max_steps::Int = 100 
    Δtrain::Int = 4 
    Δtarget_update::Int = 2000
    buffer_init::Int = max(batch_size, 200)
    log = LoggerParams(dir = "log/dqn", period = 500, rng = rng)
    device = device(π)
    buffer::ExperienceBuffer = ExperienceBuffer(sdim, adim, 1000)
    i::Int = 1
end

target(Q⁻, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q⁻(𝒟[:sp]), dims=1)

q_predicted(π, 𝒟) = sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1)

td_loss(π, 𝒟, y, L) = L(q_predicted(π, 𝒟), y)

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)

function POMDPs.solve(𝒮::DQNSolver, mdp)
    # Log the pre-train performance
    𝒮.i == 1 && log(𝒮.log, 0, mdp, 𝒮.π)
    
    # Initialize minibatch buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.sdim, 𝒮.adim, 𝒮.batch_size, device = 𝒮.device)
    γ = Float32(discount(mdp))
    s = Sampler(mdp = mdp, π = 𝒮.π, max_steps = 𝒮.max_steps, exploration_policy = 𝒮.exploration_policy, rng = 𝒮.rng)
    
    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i = 𝒮.i)
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N, step = 𝒮.Δtrain)
        # Take Δtrain steps in the environment
        push!(𝒮.buffer, steps!(s, i = 𝒮.i, Nsteps = 𝒮.Δtrain))
        
        # Sample a minibatch
        rand!(𝒮.rng, 𝒟, 𝒮.buffer, i = 𝒮.i)
        
        # Compute target, td_error and td_loss for backprop
        y = target(𝒮.π.Q⁻, 𝒟, γ)
        prioritized(𝒮.buffer) && update_priorities!(𝒮.buffer, 𝒟.indices, td_error(𝒮.π, 𝒟, y))
        loss, grad = train!(𝒮.π, () -> td_loss(𝒮.π, 𝒟, y, 𝒮.L), 𝒮.opt, 𝒮.device)
        
        # Update target network 
        elapsed(𝒮.i - 𝒮.Δtrain + 1:𝒮.i, 𝒮.Δtarget_update) && copyto!(𝒮.π.Q⁻, 𝒮.π.Q)
        
        # Log results
        log(𝒮.log, 𝒮.i - 𝒮.Δtrain + 1:𝒮.i, mdp, 𝒮.π, data = [
                                          logloss(loss),
                                          loggradient(grad),
                                          logexploration(𝒮.exploration_policy, 𝒮.i)])
    end
    𝒮.π
end

