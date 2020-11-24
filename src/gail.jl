@with_kw mutable struct GAILSolver <: Solver 
    π::DQNPolicy
    D::DQNPolicy
    N::Int64
    expert_buffer::ExperienceBuffer
    nda_buffer::Union{Nothing, ExperienceBuffer} = nothing
    λ_nda::Float32 = 1f0
    exploration_policy::ExplorationPolicy
    L::Function = Flux.Losses.huber_loss
    opt = ADAM(1e-3)
    optD = deepcopy(opt)
    batch_size::Int = 32
    target_update_period::Int = 500
    log = LoggerParams(dir = "log/gail", period = 10)
    buffer
    device = cpu
    rng::AbstractRNG = Random.GLOBAL_RNG
    i::Int64 = 1
end

function BCELoss(D, 𝒟, val::Float32)
    yh = sum(value(D, 𝒟[:s]) .* 𝒟[:a], dims = 1)
    # Flux.Losses.logitbinarycrossentropy(yh, val)
    Flux.Losses.binarycrossentropy(yh, val)
end

function Lᴰ(D, 𝒟_expert::ExperienceBuffer, 𝒟_π::ExperienceBuffer, 𝒟_nda::Union{Nothing, ExperienceBuffer}, λ_nda::Float32)
    L_e, L_π = BCELoss(D, 𝒟_expert, 1.f0), BCELoss(D, 𝒟_π, 0.f0)
    isnothing(𝒟_nda) ? L_e + L_π : L_e + λ_nda*L_π + (1.f0 - λ_nda)*BCELoss(D, 𝒟_nda, 0.f0)
end

function Lᴳ(π, D, 𝒟::ExperienceBuffer, γ::Float32, maxQ, L)
    avals = sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1) 
    target = sum(D(𝒟[:s]) .* 𝒟[:a], dims = 1) #=.+ γ .* (1f0 .- 𝒟[:done]) .* maxQ=#
    L(avals, target)
end

function POMDPs.solve(𝒮::GAILSolver, mdp)
    Q⁻, D⁻ = deepcopy(𝒮.π.Q) |> 𝒮.device, deepcopy(𝒮.D.Q) |> 𝒮.device
    buffer = ExperienceBuffer(mdp, 𝒮.buffer.size)
    fill!(buffer, mdp, RandomPolicy(mdp), 𝒮.buffer.init, rng = 𝒮.rng)
    𝒟_π = ExperienceBuffer(mdp, 𝒮.batch_size, device = 𝒮.device, Nelements = 𝒮.batch_size)
    𝒟_expert = deepcopy(𝒟_π)
    𝒟_nda = isnothing(𝒮.nda_buffer) ? nothing : deepcopy(𝒟_π)
    s, γ = rand(𝒮.rng, initialstate(mdp)) , Float32(discount(mdp))
    
    𝒮.i == 1 && log(𝒮.log, 0, mdp, 𝒮.π, rng = 𝒮.rng)
    for 𝒮.i = 𝒮.i : 𝒮.i + 𝒮.N - 1
        #TODO: Add max steps per episode
        s = push_step!(buffer, mdp, s, 𝒮.π, 𝒮.exploration_policy, 𝒮.i, rng = 𝒮.rng)
        rand!(𝒮.rng, 𝒟_π, buffer)
        rand!(𝒮.rng, 𝒟_expert, 𝒮.expert_buffer)
        !isnothing(𝒮.nda_buffer) && rand!(𝒮.rng, 𝒟_nda, 𝒮.nda_buffer)
        
        lossD, gradD = train!(𝒮.D, () -> Lᴰ(𝒮.D, 𝒟_expert, 𝒟_π, 𝒟_nda, 𝒮.λ_nda), 𝒮.optD, 𝒮.device)
        maxQ = maximum(Q⁻(𝒟_π[:sp]), dims=1)
        lossG, gradG = train!(𝒮.π, () -> Lᴳ(𝒮.π, D⁻, 𝒟_π, γ, maxQ, 𝒮.L)  +  Lᴳ(𝒮.π, D⁻, 𝒟_nda, γ, maxQ, 𝒮.L), 𝒮.opt, 𝒮.device)
        
        elapsed(𝒮.i, 𝒮.target_update_period) && begin copyto!(Q⁻, 𝒮.π.Q); copyto!(D⁻, 𝒮.D.Q) end
        log(𝒮.log, 𝒮.i, mdp, 𝒮.π, rng = 𝒮.rng, data = [logloss(lossG, gradG, suffix = "G"), 
                                                        logloss(lossD, gradD, suffix = "D"), 
                                                        logexploration(𝒮.exploration_policy, 𝒮.i)])
    end
    𝒮.π
end

