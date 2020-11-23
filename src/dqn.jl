@with_kw mutable struct DQNSolver <: Solver 
    π::DQNPolicy
    s_dim::Int
    a_dim::Int
    N::Int64 = 1000
    exploration_policy::ExplorationPolicy
    device = cpu
    L::Function = Flux.Losses.huber_loss
    opt = ADAM(1e-3)
    batch_size::Int = 32
    train_period::Int = 4 
    target_update_period::Int = 2000
    buffer_init::Int = max(batch_size, 200)
    log = LoggerParams(dir = "log/dqn", period = 500)
    buffer::ExperienceBuffer = ExperienceBuffer(mdp, 1000, device = device)
    rng::AbstractRNG = Random.GLOBAL_RNG
    i::Int64 = 1
end

target(Q⁻, 𝒟, γ) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q⁻(𝒟[:sp]), dims=1)

q_predicted(π, 𝒟) = sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1)

td_loss(π, 𝒟, y, L) = L(q_predicted(π, 𝒟), y)

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)

#TODO: Look at RL class DQN pong for inspo on gpu usage and frame processing
function POMDPs.solve(𝒮::DQNSolver, mdp; explore_offset = 0, extra_buffer = nothing)
    buffer = fill!(ExperienceBuffer, mdp, 𝒮.buffer.init, capacity = 𝒮.buffer.size, rng = 𝒮.rng)
    𝒟 = ExperienceBuffer(mdp, 𝒮.batch_size, device = 𝒮.device, Nelements = 𝒮.batch_size)
    s, γ = rand(𝒮.rng, initialstate(mdp)) , Float32(discount(mdp))
    s = StepSampler(mdp)
    
    𝒮.i == 1 && log(𝒮.log, 0, mdp, 𝒮.π, rng = 𝒮.rng)
    for i = 1:𝒮.N
        # s = push_step!(buffer, mdp, s, 𝒮.π, 𝒮.exploration_policy, 𝒮.i, rng = 𝒮.rng)
        push!(buffer, step!(sampler))
        
        rand!(𝒮.rng, 𝒟, buffer)
        y = target(Q⁻, 𝒟, γ)
        𝒮.buffer.prioritied && update_priorities!(buffer, 𝒟, td_error(𝒮.π, 𝒟, y))
        loss, grad = train!(𝒮.π, () -> td_loss(𝒮.π, 𝒟, y, 𝒮.L), 𝒮.opt, 𝒮.device)
        
        elapsed(𝒮.i, 𝒮.target_update_period) && copyto!(Q⁻, 𝒮.π.Q)
        log(𝒮.log, 𝒮.i, mdp, 𝒮.π, data = [logloss(loss, grad), logexploration(𝒮.exploration_policy, 𝒮.i)], rng = 𝒮.rng)
    end
    𝒮.π
end

# 
# function solve_multiple(𝒮::DQNSolver, mdps...; buffer = nothing)
#     mdp = mdps[1]
#     Q⁻ = deepcopy(𝒮.π.Q) |> 𝒮.device
#     𝒟 = ExperienceBuffer(mdp, 𝒮.batch_size, device = 𝒮.device, Nelements = 𝒮.batch_size)
#     svec, γ, loss, grad = [rand(𝒮.rng, initialstate(mdp)) for mdp in mdps] , Float32(discount(mdp)), NaN, NaN
# 
#     𝒮.i == 1 && log(𝒮.log, 0, mdps, 𝒮.π, rng = 𝒮.rng)
#     for 𝒮.i = 𝒮.i : 𝒮.i + 𝒮.N - 1
#         #TODO: Add max steps per episode
#         for j =1:length(mdps)
#             𝒮.π.mdp = mdps[j]
#             svec[j] = push_step!(buffer, mdps[j], svec[j], 𝒮.π, 𝒮.exploration_policy, 𝒮.i, rng = 𝒮.rng)
#         end
#         rand!(𝒮.rng, 𝒟, buffer)
# 
#         if elapsed(𝒮.i, 𝒮.train_freq)
#             y = target(Q⁻, 𝒟, γ)
#             loss, grad = train!(𝒮.π, () -> TDLoss(𝒮.π, 𝒟, y, 𝒮.L), 𝒮.opt, 𝒮.device)
#         end
# 
#         elapsed(𝒮.i, 𝒮.target_update_period) && copyto!(Q⁻, 𝒮.π.Q)
#         log(𝒮.log, 𝒮.i, mdp, 𝒮.π, data = [logloss(loss, grad), logexploration(𝒮.exploration_policy, 𝒮.i)], rng = 𝒮.rng)
#     end
#     𝒮.π
# end

