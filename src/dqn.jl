@with_kw struct DQNSolver <: Solver 
    Q
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
end

target(Q⁻, 𝒟, γ) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q⁻(𝒟[:sp]), dims=1)

TDLoss(Q, 𝒟, y, L) = L(sum(Q(𝒟[:s]) .* 𝒟[:a], dims = 1), y)


function POMDPs.solve(𝒮::DQNSolver, mdp)
    policy = CategoricalPolicy(𝒮.Q, mdp, device = 𝒮.device)
    Q⁻ = deepcopy(𝒮.Q) |> 𝒮.device
    buffer = ExperienceBuffer(mdp, RandomPolicy(mdp), 𝒮.buffer.init, 𝒮.buffer.size, rng = 𝒮.rng)
    𝒟 = ExperienceBuffer(mdp, 𝒮.batch_size, 𝒮.batch_size, device = 𝒮.device)
    s, γ = rand(𝒮.rng, initialstate(mdp)) , Float32(discount(mdp))
    
    log(𝒮.log, 0, mdp, policy, rng = 𝒮.rng)
    for i=1:𝒮.N
        s = push!(buffer, mdp, s, policy, 𝒮.exploration_policy, i, rng = 𝒮.rng)
        
        rand!(𝒮.rng, 𝒟, buffer)
        println("max reward: ", maximum(𝒟[:r]) )
        θ = Flux.params(policy, 𝒮.device)
        y = target(Q⁻, 𝒟, γ)
        Qin = network(policy, 𝒮.device)
        loss, back = Flux.pullback(() -> TDLoss(Qin, 𝒟, y, 𝒮.L), θ)
        grad = back(1f0)
        Flux.update!(𝒮.opt, θ, grad)
        sync!(policy, 𝒮.device)
        
        elapsed(i, 𝒮.target_update_period) && copyto!(Q⁻, policy.Q)
        log(𝒮.log, i, mdp, policy, data = [logloss(loss, grad, θ), logexploration(𝒮.exploration_policy, i)], rng = 𝒮.rng)
    end
    policy
end

