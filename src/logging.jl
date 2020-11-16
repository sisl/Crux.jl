function elapsed(i, N; last_i = nothing)
    isnothing(last_i) && return (i % N) == 0
    any( [last_i:i...] .% N .== 0)
end

expected_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG) = mean([simulate(RolloutSimulator(max_steps = max_steps, rng = rng), mdp, policy) for _=1:Neps])

@with_kw struct LoggerParams
    dir::String = "log/"
    period::Int64 = 100
    logger =  TBLogger(dir, tb_increment)
    eval::Function = expected_return
    performance_name::String = "undiscounted_return"
    verbose::Bool = true
end

function Base.log(p::LoggerParams, i, task, solution; data = [], rng::AbstractRNG = Random.GLOBAL_RNG, last_i = nothing)
    !elapsed(i, p.period, last_i = last_i) && return
    perf = p.eval(task, solution, rng = rng)
    log_value(p.logger, p.performance_name, perf, step = i)
    p.verbose && print("Step: $i, ", p.performance_name, ": ", perf)
    for dict in data
        for (k,v) in dict
            log_value(p.logger, k, v, step = i)
            p.verbose && print(", ", k, ": ", v)
        end
    end
    p.verbose && println()
end

# Built-in functions for logging common training things
logloss(loss, grad) = Dict("loss" => loss) # TODO Gradient
logexploration(policy, i) = policy isa EpsGreedyPolicy ? Dict("eps" => policy.eps(i)) : Dict()

