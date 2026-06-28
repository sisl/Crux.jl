using Crux
using Test
using Zygote
using Flux
using Random
using Distributions
using StatsBase
using LinearAlgebra

Random.seed!(1)

@testset "gradient penalty" begin
    m = Dense(2, 1, init = ones, bias = false)
    x = ones(Float32, 2, 100)

    @test gradient_penalty(m, x) ≈ (sqrt(2) - 1)^2
end

@testset "gradient penalty with pullback" begin
    idim = 5
    batch_size = 8

    m = Chain(Dense(idim, 2 * idim, tanh), Dense(2 * idim, 1)) |> gpu
    x = rand(Float32, idim, batch_size) |> gpu
    y = rand(Float32, 1, batch_size) |> gpu

    function total_loss()
        Flux.mse(m(x), y) + gradient_penalty(m, x)
    end

    l, b = Flux.pullback(total_loss, Flux.params(m))
    grad = b(1f0)

    @test l isa Number
    @test isfinite(l)
    @test grad.grads isa IdDict
end

@testset "GAN loss coverage" begin
    x = rand(Float32, 2, 8)
    z = rand(Float32, 2, 8)

    G = Chain(Dense(2, 2))
    D = Chain(Dense(2, 1))

    losses = [
        GAN_BCELoss(),
        GAN_LSLoss(),
        GAN_HingeLoss(),
        GAN_WLoss(),
        GAN_WLossGP()
    ]

    for loss in losses
        d_loss = Crux.Lᴰ(loss, D, x, x)
        g_loss = Crux.Lᴳ(loss, G, D, z)

        @test d_loss isa Number
        @test g_loss isa Number
        @test isfinite(d_loss)
        @test isfinite(g_loss)
    end
end

@testset "Conditional GAN loss coverage" begin
    batch_size = 8

    x = rand(Float32, 2, batch_size)
    z = rand(Float32, 2, batch_size)

    yG = [rand(Float32, 1, batch_size)]
    yD = [rand(Float32, 1, batch_size)]

    G = Chain(Dense(3, 2))
    D = Chain(Dense(3, 1))

    losses = [
        GAN_BCELoss(),
        GAN_LSLoss(),
        GAN_HingeLoss(),
        GAN_WLoss(),
        GAN_WLossGP()
    ]

    for loss in losses
        d_loss = Crux.Lᴰ(loss, D, x, x; yG = yG, yD = yD)
        g_loss = Crux.Lᴳ(loss, G, D, z; yG = yG)

        @test d_loss isa Number
        @test g_loss isa Number
        @test isfinite(d_loss)
        @test isfinite(g_loss)
    end
end

@testset "GANLosses dictionary coverage" begin
    @test haskey(GANLosses, "BCE")
    @test haskey(GANLosses, "LS")
    @test haskey(GANLosses, "Hinge")
    @test haskey(GANLosses, "W")
    @test haskey(GANLosses, "WGP")
end

@testset "ConvSN coverage" begin
    x = rand(Float32, 8, 8, 1, 4)

    layer = ConvSN((3, 3), 1 => 2, relu; pad = 1, stride = 1)

    y = layer(x)

    @test size(y) == (8, 8, 2, 4)
    @test eltype(y) <: Real
    @test all(isfinite, y)

    ps = Flux.trainable(layer)
    @test length(ps) == 2

    io = IOBuffer()
    show(io, layer)
    str = String(take!(io))

    @test occursin("ConvSN", str)
end

@testset "BatchAdjacentBuffer coverage" begin
    ref = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
    b = BatchAdjacentBuffer(elements = 3, default_z_dist = nothing, reference_buffer = ref)

    @test isprioritized(b) == false
    @test capacity(b) == 3
    @test length(b) == 0

    d1 = Dict(
        :s => ones(Float32, 2, 2),
        :a => ones(Bool, 4, 2),
        :sp => 2f0 .* ones(Float32, 2, 2),
        :r => ones(Float32, 1, 2),
        :done => zeros(Bool, 1, 2),
        :episode_end => [false true],
    )

    d2 = Dict(
        :s => 2f0 .* ones(Float32, 2, 1),
        :a => ones(Bool, 4, 1),
        :sp => 3f0 .* ones(Float32, 2, 1),
        :r => 2f0 .* ones(Float32, 1, 1),
        :done => zeros(Bool, 1, 1),
        :episode_end => [true],
    )

    d3 = Dict(
        :s => 3f0 .* ones(Float32, 2, 2),
        :a => ones(Bool, 4, 2),
        :sp => 4f0 .* ones(Float32, 2, 2),
        :r => 3f0 .* ones(Float32, 1, 2),
        :done => zeros(Bool, 1, 2),
        :episode_end => [false true],
    )

    d4 = Dict(
        :s => 4f0 .* ones(Float32, 2, 1),
        :a => ones(Bool, 4, 1),
        :sp => 5f0 .* ones(Float32, 2, 1),
        :r => 4f0 .* ones(Float32, 1, 1),
        :done => zeros(Bool, 1, 1),
        :episode_end => [true],
    )

    push!(b, d1, 0.1)
    push!(b, d2, 0.2)
    push!(b, d3, 0.3)

    @test length(b.batches) == 3
    @test length(b) == 5
    @test b.z_dists == [0.1, 0.2, 0.3]

    push!(b, d4, 0.4)

    @test length(b.batches) == 3
    @test b.z_dists[1] == 0.4
    @test b.next_ind == 2

    old_len = length(b.batches)
    push!(b, d1)
    @test length(b.batches) == old_len

    inds = get_last_N_indices(b, 2)
    @test length(inds) == 2
    @test all(1 .<= inds .<= capacity(b))
end

@testset "BatchAdjacentBuffer reservoir coverage" begin
    ref = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
    b = BatchAdjacentBuffer(elements = 2, default_z_dist = nothing, reference_buffer = ref)

    d = Dict(
        :s => ones(Float32, 2, 1),
        :a => ones(Bool, 4, 1),
        :sp => 2f0 .* ones(Float32, 2, 1),
        :r => ones(Float32, 1, 1),
        :done => zeros(Bool, 1, 1),
        :episode_end => [true],
    )

    push_reservoir!(b, d, 0.1; weight = 0.0)
    @test length(b.batches) == 0
    @test b.total_count == 0

    push_reservoir!(b, d, 0.1; weight = 1.0)
    push_reservoir!(b, d, 0.2; weight = 1.0)
    push_reservoir!(b, d, 0.3; weight = 1.0)

    @test length(b.batches) == 2
    @test b.total_count == 3
end

@testset "BatchAdjacentBuffer log averages coverage" begin
    ref = ExperienceBuffer(ContinuousSpace(2), DiscreteSpace(4), 10)
    b = BatchAdjacentBuffer(elements = 3, default_z_dist = nothing, reference_buffer = ref)

    d1 = Dict(
        :s => ones(Float32, 2, 2),
        :a => ones(Bool, 4, 2),
        :sp => 2f0 .* ones(Float32, 2, 2),
        :r => [1.0f0 3.0f0],
        :done => zeros(Bool, 1, 2),
        :episode_end => [false true],
    )

    d2 = Dict(
        :s => 2f0 .* ones(Float32, 2, 2),
        :a => ones(Bool, 4, 2),
        :sp => 3f0 .* ones(Float32, 2, 2),
        :r => [2.0f0 4.0f0],
        :done => zeros(Bool, 1, 2),
        :episode_end => [false true],
    )

    push!(b, d1, nothing)
    push!(b, d2, nothing)

    logger = log_episode_averages(b, [:r], 2)
    d = logger()

    @test haskey(d, :avg_r)
    @test d[:avg_r] ≈ 5.0
end
@testset "BatchRegularizer coverage" begin
    b = ExperienceBuffer(
        Dict(
            :s => rand(Float32, 2, 5),
            :a => rand(Float32, 1, 5),
            :value => rand(Float32, 1, 5),
        )
    )

    R = BatchRegularizer(
        buffers = [b],
        batch_size = 3,
        λ = 2f0,
        loss = (π, 𝒟) -> mean(𝒟[:value])
    )

    π = Chain(Dense(2, 1))
    val = R(π)

    @test val isa Number
    @test isfinite(val)
    @test !isnothing(R.𝒟s)
    @test length(R.𝒟s) == 1
    @test length(R.𝒟s[1]) == 3

    empty_b = ExperienceBuffer(
        Dict(
            :s => zeros(Float32, 2, 0),
            :value => zeros(Float32, 1, 0),
        )
    )

    R_empty = BatchRegularizer(
        buffers = [empty_b],
        batch_size = 3,
        λ = 2f0,
        loss = (π, 𝒟) -> error("should not run")
    )

    @test R_empty(π) == 0f0
end

@testset "DiagonalFisherRegularizer coverage" begin
    π = Chain(Dense(2, 1))
    θ = Flux.params(π)

    R = DiagonalFisherRegularizer(θ, 0.5f0)

    @test R.N == 0
    @test R.λ == 0.5f0
    @test R(π) == 0f0

    x = rand(Float32, 2, 4)
    y = rand(Float32, 1, 4)

    add_fisher_information_diagonal!(R, () -> -Flux.mse(π(x), y), θ)

    @test R.N == 1
    @test all(size(F) == size(p) for (F, p) in zip(R.F, θ))
    @test R(π) isa Number
    @test isfinite(R(π))
end

@testset "continual learning helper coverage" begin
    struct DummySolver <: Solver
        π
        log
    end

    mutable struct DummyLog
        extras
    end

    tasks = [1, 2]

    solver_generator(; kwargs...) = DummySolver(nothing, DummyLog([]))

    Crux.solve(::DummySolver, task) = nothing

    solvers = continual_learning(tasks, solver_generator)

    @test length(solvers) == 2
    @test all(s -> s isa DummySolver, solvers)
end
