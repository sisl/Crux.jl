using POMDPs
using POMDPGym
import POMDPModels
using Test
using Crux
using Flux
using Random
using BSON

Random.seed!(1)

mdp = GymPOMDP(:CartPole)
as = actions(mdp)
S = state_space(mdp)

A_analysis() = DiscreteNetwork(
    Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, length(as))),
    as
)

V_analysis() = ContinuousNetwork(
    Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 1))
)

solver_reinforce = REINFORCE(S=S, π=A_analysis())
policy_reinforce = solve(solver_reinforce, mdp)

solver_a2c = A2C(S=S, π=ActorCritic(A_analysis(), V_analysis()))
policy_a2c = solve(solver_a2c, mdp)

solver_ppo = PPO(S=S, π=ActorCritic(A_analysis(), V_analysis()))
policy_ppo = solve(solver_ppo, mdp)

@testset "Analysis learning plot coverage" begin
    p = plot_learning(
        [solver_reinforce, solver_a2c, solver_ppo],
        title = "CartPole Training Curves",
        labels = ["REINFORCE", "A2C", "PPO"]
    )

    Crux.savefig(p, "test.pdf")
    @test isfile("test.pdf")
    rm("test.pdf")
end

@testset "Analysis utility coverage" begin
    @test Crux.percentile(0.5, 10, 20) == 15
    @test Crux.percentile(0.0, 10, 20) == 10
    @test Crux.percentile(1.0, 10, 20) == 20

    @test find_crossing([1, 2, 3], [0.1, 0.7, 0.9], 0.6) == 2
    @test isnan(find_crossing([1, 2, 3], [0.1, 0.2, 0.3], 0.6))

    @test smooth([1.0, 3.0, 5.0], 0.5) == [1.0, 2.0, 3.5]

    @test directories("logs/run1") == ["logs/run1"]
    @test directories(["a", "b"]) == ["a", "b"]

    d1 = Dict(:a => [1, 2], :b => [3])
    d2 = Dict(:a => [4], :b => [5, 6])
    dcat = vcat(d1, d2)

    @test dcat[:a] == [1, 2, 4]
    @test dcat[:b] == [3, 5, 6]
end

@testset "Analysis plotting coverage" begin
    solvers = [solver_reinforce, solver_a2c, solver_ppo]

    p1 = plot_jumpstart(solvers; key = i -> :undiscounted_return)
    p2 = plot_peak_performance(solvers; key = i -> :undiscounted_return)
    p3 = plot_cumulative_rewards(
        solvers;
        key = i -> :undiscounted_return,
        show_lines = true
    )

    Crux.savefig(p1, "jumpstart.pdf")
    Crux.savefig(p2, "peak.pdf")
    Crux.savefig(p3, "cumulative_rewards.pdf")

    @test isfile("jumpstart.pdf")
    @test isfile("peak.pdf")
    @test isfile("cumulative_rewards.pdf")

    rm("jumpstart.pdf")
    rm("peak.pdf")
    rm("cumulative_rewards.pdf")
end

@testset "Analysis continual learning coverage" begin
    solvers = [solver_reinforce, solver_a2c, solver_ppo]

    x, y, breaks = Crux.cumulative_rewards(
        solvers,
        i -> :undiscounted_return
    )

    @test !isempty(x)
    @test !isempty(y)
    @test length(breaks) == length(solvers)

    res, breaks2 = Crux.single_task_performances(
        solvers,
        i -> :undiscounted_return
    )

    @test !isempty(res)
    @test length(breaks2) == length(solvers)
end

@testset "Analysis visualization coverage" begin
    frames = Crux.episode_frames(
        mdp,
        policy_ppo;
        Neps = 1,
        max_steps = 2,
        use_obs = false
    )

    @test frames isa Vector
    @test !isempty(frames)

    Crux.gif(frames, "cartpole.gif"; fps = 5)

    @test isfile("cartpole.gif")

    rm("cartpole.gif")
end

@testset "Analysis tb2dict coverage" begin
    dirs = directories([solver_reinforce, solver_a2c, solver_ppo])
    @test !isempty(dirs)

    try
        d = Crux.tb2dict(dirs[1], [:undiscounted_return])
        @test haskey(d, :iterations)
    catch e
        @test e isa Exception
    end
end

@testset "Analysis extra plot coverage" begin
    solvers = [solver_reinforce, solver_a2c, solver_ppo]

    p1 = Crux.plot_steps_to_threshold(
        solvers,
        -Inf;
        key = i -> :undiscounted_return
    )

    p3 = plot_learning(
        [solver_reinforce];
        values = :undiscounted_return,
        labels = :default
    )

    p4 = plot_learning(
        solvers;
        values = :undiscounted_return,
        labels = :default
    )

    for (p, f) in zip(
        [p1, p3, p4],
        ["threshold.pdf", "learning_one.pdf", "learning_multi.pdf"]
    )
        Crux.savefig(p, f)
        @test isfile(f)
        rm(f)
    end

    try
        p2 = Crux.plot_forgetting(
            solvers;
            key = i -> :undiscounted_return
        )

        Crux.savefig(p2, "forgetting.pdf")
        @test isfile("forgetting.pdf")
        rm("forgetting.pdf")
    catch e
        @test e isa KeyError
    end
end

@testset "Analysis observation frame coverage" begin
    frames = Crux.episode_frames(
        mdp,
        policy_ppo;
        Neps = 1,
        max_steps = 2,
        use_obs = true
    )

    @test frames isa Vector
    @test !isempty(frames)
end
