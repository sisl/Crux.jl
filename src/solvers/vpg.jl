@with_kw mutable struct VPGSolver <: Solver 
    π::Policy
    sdim::Int
    adim::Int
    baseline::Baseline
    N::Int64 = 1000
    ΔN::Int = 1000
    batch_size::Int = 32
    max_steps::Int64 = 100
    opt = ADAM(1e-3)
    device = device(π)
    rng::AbstractRNG = Random.GLOBAL_RNG
    log = LoggerParams(dir = "log/vpg", period = 500)
    i::Int64 = 0
end

vpg_loss(π, 𝒟) = -mean(logpdf(π, 𝒟[:s], 𝒟[:a]) .* 𝒟[:advantage])

function POMDPs.solve(𝒮::VPGSolver, mdp)
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_discounted_return(mdp, 𝒮.π, 𝒮.rng))
    
    # Construct the experience buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.sdim, 𝒮.adim, 𝒮.ΔN, device = 𝒮.device, gae = true)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, max_steps = 𝒮.max_steps, rng = 𝒮.rng)
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.ΔN, step = 𝒮.ΔN)
        # Sample transitions
        push!(𝒟, steps!(s, Nsteps = 𝒮.ΔN, baseline = 𝒮.baseline, γ = γ, reset = true))
        
        # Train the baseline
        train!(𝒮.baseline, 𝒟)
        
        # Train the policy (using batches)
        losses, grads = train!(𝒮.π, (D) -> vpg_loss(𝒮.π, D), 𝒟, 𝒮.batch_size, 𝒮.opt, 𝒮.device, rng = 𝒮.rng)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_discounted_return(mdp, 𝒮.π, 𝒮.rng), 
                                        log_loss(losses),
                                        log_gradient(grads))
    end
    𝒮.i += 𝒮.ΔN
end

