using Crux
using Test
using Flux
using Random

Random.seed!(1)

@testset "ExperienceReplay constructor coverage" begin
    S = ContinuousSpace(2)
    A = ContinuousSpace(1)

    π = ActorCritic(
        ContinuousNetwork(Chain(Dense(2, 1)), A),
        DoubleNetwork(
            ContinuousNetwork(Chain(Dense(3, 1))),
            ContinuousNetwork(Chain(Dense(3, 1)))
        )
    )

    solver = ExperienceReplay(
        π = π,
        S = S,
        A = A,
        N_experience_replay = 10,
        buffer_size = 20,
        ΔN = 2,
        N = 2,
        solver = TD3
    )

    @test solver isa OffPolicySolver
    @test haskey(solver.𝒫, :buffer_er)
end

@testset "TIER loss helper coverage" begin
    S = ContinuousSpace(2)
    A = ContinuousSpace(1)

    π = ActorCritic(
        ContinuousNetwork(Chain(Dense(3, 1)), A),
        DoubleNetwork(
            ContinuousNetwork(Chain(Dense(4, 1))),
            ContinuousNetwork(Chain(Dense(4, 1)))
        )
    )

    𝒟 = Dict(
        :s => rand(Float32, 2, 4),
        :sp => rand(Float32, 2, 4),
        :a => rand(Float32, 1, 4),
        :r => rand(Float32, 1, 4),
        :done => zeros(Float32, 1, 4),
        :z => rand(Float32, 1, 4),
        :value => rand(Float32, 1, 4),
        :weight => ones(Float32, 1, 4)
    )

    y = rand(Float32, 1, 4)
    info = Dict()

    td_loss = Crux.TIER_td_loss()
    loss_val = td_loss(critic(π).N1, (;), 𝒟, y; info = info, z = 𝒟[:z])

    @test loss_val isa Number
    @test isfinite(loss_val)
    @test haskey(info, :Qavg)

    double_loss = Crux.TIER_double_Q_loss()
    double_val = double_loss(π, (;), 𝒟, y; info = Dict(), z = 𝒟[:z])

    @test double_val isa Number
    @test isfinite(double_val)

    actor_loss = Crux.TIER_TD3_actor_loss(π, (;), 𝒟)

    @test actor_loss isa Number
    @test isfinite(actor_loss)

    areg = Crux.TIER_action_regularization(π, 𝒟)
    vreg = Crux.TIER_action_value_regularization(π, 𝒟)

    @test areg isa Number
    @test vreg isa Number
    @test isfinite(areg)
    @test isfinite(vreg)
end

@testset "TIER constructor coverage" begin
    S = ContinuousSpace(2)
    A = ContinuousSpace(1)

    latent_dim = 1

    π = ActorCritic(
        ContinuousNetwork(Chain(Dense(2 + latent_dim, 1)), A),
        DoubleNetwork(
            ContinuousNetwork(Chain(Dense(2 + latent_dim + 1, 1))),
            ContinuousNetwork(Chain(Dense(2 + latent_dim + 1, 1)))
        )
    )

    observation_model = Chain(Dense(2 + 1, 4, relu), Dense(4, latent_dim))

    obs_loss(model, D; kwargs...) = Flux.mse(model(vcat(D[:s], D[:a])), D[:z])

    solver = TIER(
        π = π,
        observation_model = observation_model,
        S = S,
        A = A,
        ΔN = 2,
        latent_dim = latent_dim,
        N_experience_replay = 10,
        N_experience_obs = 10,
        buffer_size = 20,
        bayesian_inference = (model, D, prior; info = Dict()) -> prior,
        obs_opt = (; loss = obs_loss, epochs = 0, batch_size = 2),
        solver = TD3,
        N = 2
    )

    @test solver isa OffPolicySolver
    @test haskey(solver.𝒫, :buffer_er)
    @test haskey(solver.𝒫, :buffer_obs)
    @test haskey(solver.𝒫, :observation_model)
    @test haskey(solver.𝒫, :z_dist)
    @test haskey(solver.𝒫, :zs)
end
