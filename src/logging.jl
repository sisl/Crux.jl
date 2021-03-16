elapsed(i::Int, N::Int) = (i % N) == 0
elapsed(i::UnitRange, N::Int) = any([i...] .% N .== 0)

@with_kw mutable struct LoggerParams
    dir::String = "log/"
    period::Int64 = 500
    logger = TBLogger(dir, tb_increment)
    fns = [log_undiscounted_return(10)]
    verbose::Bool = true
end

Base.log(p::Nothing, i, data...; kwargs...)  = nothing

#Note that i can be an int or a unitrange
function Base.log(p::LoggerParams, i::Union{Int, UnitRange}, data...; s = nothing)
    !elapsed(i, p.period) && return
    i = i[end]
    p.verbose && print("Step: $i")
    for dict in [p.fns..., data..., log_exploration(s.π_explore)]
        d = dict isa Function ? dict(s=s, i=i) : dict
        for (k,v) in d
            log_value(p.logger, string(k), v, step = i)
            p.verbose && print(", ", k, ": ", v)
        end
    end
    p.verbose && println()
end
# @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e | EvalR %1.3f \n",
                        #t, solver.max_steps, nt[1], avg100_reward, loss_val, grad_val, scores_eval)
                        
function aggregate_info(infos)
    res = Dict(k => mean([info[k] for info in filter((x)->haskey(x, k), infos)]) for k in keys(infos[1]))
end

# Built-in functions for logging common training things
log_performance(s::AbstractVector, name, fn; kwargs...) = Dict("$(name)/T$i" => fn(s[i]; kwargs...) for i=1:length(s))
log_performance(s::Sampler, name, fn; kwargs...) = Dict(name => fn(s; kwargs...))

log_discounted_return(Neps) = (;s, kwargs...) -> log_performance(s, "discounted_return", discounted_return, Neps=Neps)
log_undiscounted_return(Neps; name="undiscounted_return") = (;s, kwargs...) -> log_performance(s, name, undiscounted_return, Neps=Neps)
log_undiscounted_return(s, Neps; name="undiscounted_return") = (;kwargs...) -> log_performance(s, name, undiscounted_return, Neps=Neps)
log_failure(Neps) = (;s, kwargs...) -> log_performance(s, "failure_rate", failure, Neps=Neps)

log_exploration(policy) = (;kwargs...) -> Dict()
log_exploration(policy::ϵGreedyPolicy; name = "eps") = (;i, kwargs...) -> Dict(name => policy.ϵ(i))
log_exploration(policy::GaussianNoiseExplorationPolicy; name = "noise_std") = (;i, kwargs...) -> Dict(name => policy.σ(i))
function log_exploration(policy::FirstExplorePolicy; name = "first_explore_on")
    (;i, kwargs...) -> begin
        d = Dict{String, Any}(name => i < policy.N)
        !isnothing(policy.after_policy) && merge!(d, log_exploration(policy.after_policy)(;i=1,kwargs...))
        d
    end
end

