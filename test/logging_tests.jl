using Crux
using Test

@testset "logging callback coverage" begin
    @test log_discounted_return(1) isa Function
    @test log_undiscounted_return(1) isa Function
    @test log_failure(1) isa Function
    @test log_metric_by_key(:reward, 1) isa Function
    @test log_metrics_by_key([:reward], 1) isa Function
    @test log_episode_averages([:reward], 2) isa Function
    @test log_experience_sums([:reward], 2) isa Function
end

@testset "save_gif early return coverage" begin
    cb = save_gif(log_at_zero = false)

    s = (
        dir = tempdir(),
        mdp = nothing,
        agent = (; π = nothing)
    )

    @test cb(i = 0, s = s, dir = tempdir(), logger = nothing) === nothing
end
