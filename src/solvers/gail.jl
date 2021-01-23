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

function dqn_Lᴰ(D, 𝒟_expert, 𝒟_π, no_nda::Nothing, λ_nda::Float32; info = Dict())
    LBCE(value(D, 𝒟_expert[:s], 𝒟_expert[:a]), 1.f0) + LBCE(value(D, 𝒟_π[:s], 𝒟_π[:a]), 0.f0)
end

function dqn_Lᴰ(D, 𝒟_expert, 𝒟_π, 𝒟_nda, λ_nda::Float32; info = Dict())
    LBCE(value(D, 𝒟_expert[:s], 𝒟_expert[:a]), 1.f0) + 
    LBCE(value(D, 𝒟_π[:s], 𝒟_π[:a]), 0.f0) + 
    λ_nda*LBCE(value(D, 𝒟_nda[:s], 𝒟_nda[:a]), 0.f0)
end

## DQN-GAIL stuff
dqngail_target(π, D, 𝒟, γ::Float32) = tanh.(value(D, 𝒟[:s], 𝒟[:a])) .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(value(π, 𝒟[:sp]), dims=1)

function POMDPs.solve(𝒮GAIL::GAILSolver{DQNSolver}, mdp)    
    𝒮 = 𝒮GAIL.G # pull out the main solver
    @assert !(prioritized(𝒮.buffer)) # not handled
    
    # Initialize minibatch buffers and sampler
    𝒟_π = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.batch_size, device = 𝒮.device)
    𝒟_expert = deepcopy(𝒟_π)
    𝒟_nda = isnothing(𝒮GAIL.nda_buffer) ? nothing : deepcopy(𝒟_π)
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.π, 𝒮.S, max_steps = 𝒮.max_steps, exploration_policy = 𝒮.exploration_policy, rng = 𝒮.rng)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    # Fill the buffer as needed
    𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i = 𝒮.i, explore = true)
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.ΔN, step = 𝒮.ΔN)
        # Take ΔN steps in the environment
        push!(𝒮.buffer, steps!(s, explore = true, i = 𝒮.i, Nsteps = 𝒮.ΔN))
        
        infos = []
        for _ in 1:𝒮.epochs
            # Sample a minibatch
            rand!(𝒮.rng, 𝒟_π, 𝒮.buffer, i = 𝒮.i)
            rand!(𝒮.rng, 𝒟_expert, 𝒮GAIL.expert_buffer, i = 𝒮.i)
            !isnothing(𝒮GAIL.nda_buffer) && rand!(𝒮.rng, 𝒟_nda, 𝒮GAIL.nda_buffer, i = 𝒮.i)
            
            # Train the discriminator
            info_D = train!(𝒮GAIL.D, 
                            (;kwargs...) -> dqn_Lᴰ(𝒮GAIL.D, 𝒟_expert, 𝒟_π, 𝒟_nda, 𝒮GAIL.λ_nda; kwargs...), 
                            𝒮GAIL.optD, 
                            loss_sym = :loss_D, 
                            grad_sym = :grad_norm_D)
            
            # Compute target and train the generato
            y = dqngail_target(𝒮.π⁻, 𝒮GAIL.D, 𝒟_π, γ)
            info_G = train!(𝒮.π, 
                            (;kwargs...) -> td_loss(𝒮.π, 𝒟_π, y, 𝒮.loss; kwargs...), 
                            𝒮.opt, loss_sym = :loss_G, 
                            grad_sym = :grad_norm_G)
            
            push!(infos, merge(info_D, info_G))
        end
        # Update target network
        elapsed(𝒮.i + 1:𝒮.i + 𝒮.ΔN, 𝒮.Δtarget_update) && copyto!(𝒮.π⁻, 𝒮.π)
        
        # Log results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                            aggregate_info(infos),
                                            log_exploration(𝒮.exploration_policy, 𝒮.i))
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

## PG-GAIL stuff
function pg_Lᴰ(D, 𝒟_expert, 𝒟_π; info = Dict())
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
    s = Sampler(mdp, 𝒮.π, 𝒮.S, required_columns = 𝒮.required_columns, λ = 𝒮.λ_gae, max_steps = 𝒮.max_steps, rng = 𝒮.rng, exploration_policy = 𝒮.π)
    
    # Log the pre-train performance
    𝒮.i == 0 && log(𝒮.log, 𝒮.i, log_undiscounted_return(s, Neps = 𝒮.eval_eps))
    
    for 𝒮.i = range(𝒮.i, stop = 𝒮.i + 𝒮.N - 𝒮.ΔN, step = 𝒮.ΔN)
        # Sample transitions
        push!(𝒟, steps!(s, Nsteps = 𝒮.ΔN, reset = true, explore = true))
        
        # Train the discriminator (using batches)
        if isnothing(𝒮GAIL.nda_buffer)
            info_D = train!(𝒮GAIL.D, pg_Lᴰ, 𝒮.batch_size, 𝒮GAIL.optD, 
                                  𝒮GAIL.expert_buffer, 𝒟,
                                  epochs = 𝒮.epochs, rng = 𝒮.rng,
                                  loss_sym = :loss_D, grad_sym = :grad_norm_D)
        else
            #TODO
            error("not implemented")
        end
        
        𝒟[:advantage] .= value(𝒮GAIL.D, vcat(𝒟[:s], 𝒟[:a]))
        
        # Normalize the advantage
        𝒮.normalize_advantage && (𝒟[:advantage] .= whiten(𝒟[:advantage]))
        
        # Train the policy (using batches)
        info_G = train!(𝒮.π, 𝒮.loss, 𝒮.batch_size, 𝒮.opt, 𝒟, 
                        epochs = 𝒮.epochs, 
                        rng = 𝒮.rng, 
                        regularizer = 𝒮.regularizer, 
                        early_stopping = 𝒮.early_stopping,
                        loss_sym = :policy_loss_G,
                        grad_sym = :policy_grad_norm_G)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, log_undiscounted_return(s, Neps = 𝒮.eval_eps), 
                                        info_D, 
                                        info_G)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end


