@with_kw mutable struct GAILSolver{T} <: Solver
    D
    G::T
    optD = deepcopy(G.opt)
    expert_buffer::ExperienceBuffer
    nda_buffer::Union{Nothing, ExperienceBuffer} = nothing
    λ_nda::Float32 = 0.5f0
end

## Discriminator stuff
const LBCE = Flux.Losses.logitbinarycrossentropy

function dqn_Lᴰ(D, 𝒟_expert, 𝒟_π)
    LBCE(q_predicted(D, 𝒟_expert), 1.f0) + LBCE(q_predicted(D, 𝒟_π), 0.f0)
end

function dqn_Lᴰ_nda(D, 𝒟_expert, 𝒟_π, 𝒟_nda, λ_nda::Float32)
    LBCE(q_predicted(D, 𝒟_expert), 1.f0) +  LBCE(q_predicted(D, 𝒟_π), 0.f0) + λ_nda*LBCE(q_predicted(D, 𝒟_nda), 0.f0)
end

## DQN-GAIL stuff
dqngail_target(Q, D, 𝒟, γ::Float32) = tanh.(q_predicted(D, 𝒟)) .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q(𝒟[:sp]), dims=1)

function POMDPs.solve(𝒮GAIL::GAILSolver{DQNSolver}, mdp)
    𝒮 = 𝒮GAIL.G # pull out the main solver
    
    # Initialize minibatch buffers and sampler
    𝒟_π = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.batch_size, device = 𝒮.device)
    𝒟_expert = deepcopy(𝒟_π)
    𝒟_nda = isnothing(𝒮GAIL.nda_buffer) ? nothing : deepcopy(𝒟_π)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, max_steps = 𝒮.max_steps, exploration_policy = 𝒮.exploration_policy, rng = 𝒮.rng)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i = 𝒮.i)
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.Δtrain, step = 𝒮.Δtrain)
        # Take Δtrain steps in the environment
        push!(𝒮.buffer, steps!(s, i = 𝒮.i, Nsteps = 𝒮.Δtrain))
        
        # Sample a minibatch
        rand!(𝒮.rng, 𝒟_π, 𝒮.buffer, i = 𝒮.i)
        rand!(𝒮.rng, 𝒟_expert, 𝒮GAIL.expert_buffer, i = 𝒮.i)
        !isnothing(𝒮GAIL.nda_buffer) && rand!(𝒮.rng, 𝒟_nda, 𝒮GAIL.nda_buffer, i = 𝒮.i)
        
        # train the discrimnator
        if isnothing(𝒮GAIL.nda_buffer)
            lossD, gradD = train!(𝒮GAIL.D, () -> dqn_Lᴰ(𝒮GAIL.D, 𝒟_expert, 𝒟_π), 𝒮GAIL.optD)
        else
            lossD, gradD = train!(𝒮GAIL.D, () -> dqn_Lᴰ_nda(𝒮GAIL.D, 𝒟_expert, 𝒟_π, 𝒟_nda, 𝒮GAIL.λ_nda), 𝒮GAIL.optD)
        end
        
        # Compute target, update priorities, and train the generator.
        y = dqngail_target(𝒮.π.Q⁻, 𝒮GAIL.D, 𝒟_π, γ)
        prioritized(𝒮.buffer) && update_priorities!(𝒮.buffer, 𝒟_π.indices, td_error(𝒮.π, 𝒟_π, y))
        lossG, gradG = train!(𝒮.π, () -> td_loss(𝒮.π, 𝒟_π, y, 𝒮.L), 𝒮.opt)
            
        # Update target network
        elapsed(𝒮.i + 1:𝒮.i + 𝒮.Δtrain, 𝒮.Δtarget_update) && copyto!(𝒮.π.Q⁻, 𝒮.π.Q)
        
        # Log results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.Δtrain, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                            log_loss(lossG, suffix = "G"),
                                            log_loss(lossD, suffix = "D"),
                                            log_gradient(gradG, suffix = "G"),
                                            log_gradient(gradD, suffix = "D"),
                                            log_exploration(𝒮.exploration_policy, 𝒮.i))
    end
    𝒮.i += 𝒮.Δtrain
    𝒮.π
end

## PG-GAIL stuff
function pg_Lᴰ(D, 𝒟_expert, 𝒟_π)
    LBCE(value(D, vcat(𝒟_expert[:s], 𝒟_expert[:a])), 1.f0) + LBCE(value(D, vcat(𝒟_π[:s], 𝒟_π[:a])), 0.f0)
end

# function pg_Lᴰ_nda(D, 𝒟_expert, 𝒟_π, 𝒟_nda, λ_nda::Float32)
#     LBCE(q_predicted(D, 𝒟_expert), 1.f0) +  LBCE(q_predicted(D, 𝒟_π), 0.f0) + λ_nda*LBCE(q_predicted(D, 𝒟_nda), 0.f0)
# end

function POMDPs.solve(𝒮GAIL::GAILSolver{PGSolver}, mdp)
    𝒮 = 𝒮GAIL.G # pull out the main solver
    
    # Construct the experience buffer and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device = 𝒮.device)
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae
    s = Sampler(mdp, 𝒮.π, 𝒮.S, 𝒮.A, required_columns = 𝒮.required_columns, λ = 𝒮.λ_gae, max_steps = 𝒮.max_steps, rng = 𝒮.rng)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.ΔN, step = 𝒮.ΔN)
        # Sample transitions
        push!(𝒟, steps!(s, Nsteps = 𝒮.ΔN, reset = true))
        
        # Train the discriminator (using batches)
        if isnothing(𝒮GAIL.nda_buffer)
            lossD, gradD = train!(𝒮GAIL.D, 
                                  (Dexp, Dπ) -> pg_Lᴰ(𝒮GAIL.D, Dexp, Dπ), 
                                  𝒮.batch_size, 𝒮GAIL.optD, 
                                  𝒮GAIL.expert_buffer, 𝒟,
                                  epochs = 𝒮.epochs, rng = 𝒮.rng)
        else
            error("not implemented")
            # lossD, gradD = train!(𝒮GAIL.D, 
                                  # (Dexp, Dπ, Dnda) -> Lᴰ_nda(𝒮GAIL.D, Dexp, Dπ, Dnda, 𝒮GAIL.λ_nda), 
                                  # 𝒮.batch_size, 𝒮GAIL.optD, 
                                  # 𝒮GAIL.expert_buffer, 𝒟, 𝒮GAIL.nda_buffer, 
                                  # epochs = 𝒮.epochs, rng = 𝒮.rng)
        end
        
        𝒟[:advantage] .= value(𝒮GAIL.D, vcat(𝒟[:s], 𝒟[:a]))
            
        
        # Train the policy (using batches)
        losses, grads = train!(𝒮.π, (D) -> 𝒮.loss(𝒮.π, D), 𝒮.batch_size, 𝒮.opt, 𝒟, epochs = 𝒮.epochs, rng = 𝒮.rng)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                        log_loss(losses, suffix = "G"),
                                        log_gradient(grads, suffix = "G"),
                                        log_loss(lossD, suffix = "D"),
                                        log_gradient(gradD, suffix = "D"),)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end


