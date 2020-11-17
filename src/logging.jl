function elapsed(i, N; last_i = nothing)
    isnothing(last_i) && return (i % N) == 0
    any( [last_i:i...] .% N .== 0)
end

discounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG) = mean([simulate(RolloutSimulator(max_steps = max_steps, rng = rng), mdp, policy) for _=1:Neps])
undiscounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG) = mean([undiscounted_reward(simulate(HistoryRecorder(max_steps = max_steps, rng = rng), mdp, policy)) for _=1:Neps])
failure_rate(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG) = mean([simulate(RolloutSimulator(max_steps = max_steps, rng = rng), mdp, policy) < 0. for _=1:Neps])

@with_kw struct LoggerParams
    dir::String = "log/"
    period::Int64 = 100
    logger =  TBLogger(dir, tb_increment)
    eval::Function = (task, solution; kwargs...) -> Dict("discounted_return" => discounted_return(task, solution; kwargs...))
    other::Function = (p) -> Dict()
    verbose::Bool = true
end

function Base.log(p::LoggerParams, i, task, solution; data = [], rng::AbstractRNG = Random.GLOBAL_RNG, last_i = nothing)
    !elapsed(i, p.period, last_i = last_i) && return
    perf = p.eval(task, solution, rng = rng)
    p.verbose && print("Step: $i, ")
    for dict in [perf, data..., p.other(solution)]  
        for (k,v) in dict
            log_value(p.logger, k, v, step = i)
            p.verbose && print(", ", k, ": ", v)
        end
    end
    p.verbose && println()
end

# Built-in functions for logging common training things
logloss(loss, grad; name = "loss", suffix = "") = Dict(string(name, suffix) => loss) # TODO Gradient
logexploration(policy, i; name = "eps") = policy isa EpsGreedyPolicy ? Dict(name => policy.eps(i)) : Dict()

