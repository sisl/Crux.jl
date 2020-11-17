@with_kw mutable struct DQNSolver <: Solver 
    π::DQNPolicy
    N::Int64
    exploration_policy::ExplorationPolicy
    L::Function = Flux.Losses.huber_loss
    opt = ADAM(1e-3)
    batch_size::Int = 32
    target_update_period::Int = 2000
    log = LoggerParams(dir = "log/dqn", period = 500)
    buffer::BufferParams = BufferParams(init = 200, size = 1000)
    device = cpu
    rng::AbstractRNG = Random.GLOBAL_RNG
    i::Int64 = 1
end

target(Q⁻, 𝒟, γ) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q⁻(𝒟[:sp]), dims=1)

TDLoss(π, 𝒟, y, L) = L(sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1), y)


function POMDPs.solve(𝒮::DQNSolver, mdp)
    Q⁻ = deepcopy(𝒮.π.Q) |> 𝒮.device
    buffer = ExperienceBuffer(mdp, 𝒮.buffer.size)
    fill!(buffer, mdp, RandomPolicy(mdp), 𝒮.buffer.init, rng = 𝒮.rng)
    𝒟 = ExperienceBuffer(mdp, 𝒮.batch_size, device = 𝒮.device, Nelements = 𝒮.batch_size)
    s, γ = rand(𝒮.rng, initialstate(mdp)) , Float32(discount(mdp))
    
    𝒮.i == 1 && log(𝒮.log, 0, mdp, 𝒮.π, rng = 𝒮.rng)
    for 𝒮.i = 𝒮.i : 𝒮.i + 𝒮.N - 1
        #TODO: Add max steps per episode
        s = push_step!(buffer, mdp, s, 𝒮.π, 𝒮.exploration_policy, 𝒮.i, rng = 𝒮.rng)
        rand!(𝒮.rng, 𝒟, buffer)
        
        y = target(Q⁻, 𝒟, γ)
        loss, grad = train!(𝒮.π, () -> TDLoss(𝒮.π, 𝒟, y, 𝒮.L), 𝒮.opt, 𝒮.device)
        
        elapsed(𝒮.i, 𝒮.target_update_period) && copyto!(Q⁻, 𝒮.π.Q)
        log(𝒮.log, 𝒮.i, mdp, 𝒮.π, data = [logloss(loss, grad), logexploration(𝒮.exploration_policy, 𝒮.i)], rng = 𝒮.rng)
    end
    𝒮.π
end

