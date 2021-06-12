@with_kw mutable struct ASAFSolver <: Solver
    π # Policy
    S::AbstractSpace # State space
    A::AbstractSpace = action_space(π) # Action space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 2000 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = LoggerParams(;dir = "log/ASAF") # The logging parameters
    i::Int = 0 # The current number of environment interactions
    a_opt::TrainingParams # Training parameters for the actor
    required_columns = Symbol[]
    𝒟_demo
end

function ASAF(;π, S, A=action_space(π), 𝒟_demo, normalize_demo::Bool=true, ΔN=50, λ_orth=1f-4, a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> device(π)
    ASAFSolver(;π=π, 
                 S=S, 
                 A=A,
                 𝒟_demo=𝒟_demo,
                 ΔN=ΔN,
                 log=LoggerParams(;dir="log/ASAF", period=100, log...),
                 a_opt=TrainingParams(;name="actor_", loss=nothing, a_opt...), 
                 kwargs...)
end


function ASAF_actor_loss(πG)
    (π, 𝒟, 𝒟_demo; info=Dict()) -> begin
        πsa_G = logpdf(π, 𝒟[:s], 𝒟[:a])
        πsa_E = logpdf(π, 𝒟_demo[:s], 𝒟_demo[:a])
        πGsa_G = logpdf(πG, 𝒟[:s], 𝒟[:a])
        πGsa_E = logpdf(πG, 𝒟_demo[:s], 𝒟_demo[:a])
        e = mean(entropy(π, 𝒟[:s]))
        
        ignore() do
            info[:entropy] = e
        end 
        
        L = Flux.mean(log.(1 .+ exp.(πGsa_E - πsa_E))) + Flux.mean(log.(exp.(πsa_G - πGsa_G)  .+ 1))  - 0.1f0*e
        # if !isnothing(𝒟_nda)
        #     πsa_NDA = logpdf(π, 𝒟_nda[:s], 𝒟_nda[:a])
        #     πGsa_NDA = logpdf(πG, 𝒟_nda[:s], 𝒟_nda[:a])
        #     L += Flux.mean(log.(1 .+ exp.(πsa_NDA - πGsa_NDA)))
        # end
        L
    end
end

function POMDPs.solve(𝒮::ASAFSolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.A, 𝒮.ΔN, 𝒮.required_columns, device=device(𝒮.π))
    s = Sampler(mdp, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π, required_columns=𝒮.required_columns)
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Log the pre-train performance
    log(𝒮.log, 𝒮.i)

    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Sample transitions into the batch buffer
        push!(𝒟, steps!(s, Nsteps=𝒮.ΔN, reset=true, explore=true, i=𝒮.i))
        
        # Train the actor
        𝒮.a_opt.loss = ASAF_actor_loss(deepcopy(𝒮.π))
        info = batch_train!(actor(𝒮.π), 𝒮.a_opt, 𝒟, 𝒮.𝒟_demo)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, info)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.π
end

