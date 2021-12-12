elapsed(i::Int, N::Int) = (i % N) == 0
elapsed(i::UnitRange, N::Int) = any([i...] .% N .== 0)

function TensorBoardLogger.log_value(logger::WBLogger, name::AbstractString, value::Real; step=0)
    info_dict = Dict(string(name) => value)
    wandb.log(info_dict, step=step)
end

@with_kw mutable struct LoggerParams
    dir::String = "log/"
    period::Int64 = 500
    use_wandb::Bool = false
    config::Union{Dict, Nothing} = nothing
    project::Union{AbstractString, Nothing} = nothing
    entity::Union{AbstractString, Nothing} = nothing
    logger::Union{TBLogger, WBLogger} = use_wandb ? WBLogger(name=dir, config=config, project=project, entity=entity) : TBLogger(dir, tb_increment)
    fns = Any[log_undiscounted_return(10)]
    writeout::Dict{Int, Any} = Dict() # Other things to write to disk. Period => Function
    verbose::Bool = true
    sampler::Union{Sampler, Nothing, Vector} = nothing
end

Base.log(p::Nothing, i, data...; kwargs...)  = nothing

#Note that i can be an int or a unitrange
function Base.log(p::LoggerParams, i::Union{Int, UnitRange}, data...)
    
    # Write things to disc
    if p.logger isa TBLogger
        for (period, fn) in p.writeout
            elapsed(i, period) && fn(i=i, s=p.sampler, dir=p.logger.logdir)
        end
    end
    
    # Save other run information
    !elapsed(i, p.period) && return
    i = i[end]
    p.verbose && print("Step: $i")
    π_explore = (p.sampler isa Vector) ?  first(p.sampler).agent.π_explore : p.sampler.agent.π_explore
        
    for dict in [p.fns..., data..., log_exploration(π_explore)]
        d = dict isa Function ? dict(s=p.sampler, i=i) : dict
        for (k,v) in d
            p.verbose && print(", ", k, ": ", v)
            log_value(p.logger, string(k), v, step = i)
        end
    end
    p.verbose && println()
end
# @printf("%5d / %5d eps %0.3f |  avgR %1.3f | Loss %2.3e | Grad %2.3e | EvalR %1.3f \n",
                        #t, solver.max_steps, nt[1], avg100_reward, loss_val, grad_val, scores_eval)
                        
function aggregate_info(infos)
    res = Dict()
    for k in unique(vcat([collect(keys(info)) for info in infos]...))
        res[k] = mean([info[k] for info in filter((x)->haskey(x, k), infos)])
    end
    res
end

# Built-in functions for logging common training things
log_performance(s::AbstractVector, name, fn; kwargs...) = Dict("$(name)/T$i" => fn(s[i]; kwargs...) for i=1:length(s))
log_performance(s::Sampler, name, fn; kwargs...) = Dict(name => fn(s; kwargs...))

log_discounted_return(Neps) = (;s, kwargs...) -> log_performance(s, "discounted_return", discounted_return, Neps=Neps)
log_undiscounted_return(Neps; name="undiscounted_return") = (;s, kwargs...) -> log_performance(s, name, undiscounted_return, Neps=Neps)
log_undiscounted_return(s, Neps; name="undiscounted_return") = (;kwargs...) -> log_performance(s, name, undiscounted_return, Neps=Neps)
log_failure(Neps) = (;s, kwargs...) -> log_performance(s, "failure_rate", failure, Neps=Neps)
log_metric_by_key(key, Neps) = (;s, kwargs...) -> log_performance(s, string(key), metric_by_key, key=key, Neps=Neps)

function log_metrics_by_key(keys, Neps; kwargs...)
    (;s,kwargs2...) -> begin
        vals = metrics_by_key(s, keys=keys, Neps=Neps, kwargs...)
        Dict(k=>v for (k,v) in zip(keys, vals))
    end
end

log_validation_error(loss, 𝒟_val; name="validation_error") = (;s, kwargs...) -> Dict(name => loss(s.π, 𝒟_val))

log_exploration(policy) = (;kwargs...) -> Dict()
log_exploration(policy::MixedPolicy; name = "eps") = (;i, kwargs...) -> Dict(name => policy.ϵ(i))
log_exploration(policy::GaussianNoiseExplorationPolicy; name = "noise_std") = (;i, kwargs...) -> Dict(name => policy.σ(i))
function log_exploration(policy::FirstExplorePolicy; name = "first_explore_on")
    (;i, kwargs...) -> begin
        d = Dict{String, Any}(name => i < policy.N)
        !isnothing(policy.after_policy) && merge!(d, log_exploration(policy.after_policy)(;i=1,kwargs...))
        d
    end
end


function log_episode_averages(buffer::ExperienceBuffer, keys, period)
    (;kwargs...) -> begin
        d = Dict()
        indices = get_last_N_indices(buffer, period)
        for k in keys
            avg_val = sum(buffer[k][indices]) / sum(buffer[:episode_end][indices])
            d[Symbol(string("avg_", k))] = avg_val
        end
        d
    end
end

function log_experience_sums(buffer, keys, period)
    (;kwargs...) -> begin
        d = Dict()
        indices = get_last_N_indices(buffer, period)
        for k in keys
            avg_val = sum(buffer[k][indices])
            d[Symbol(string("sum_", k))] = avg_val
        end
        d
    end
end

function save_gif(;Neps=1, base_name="demo", log_at_zero = false)
    (;i, s, dir) -> begin
        !log_at_zero && i==0 && return
        filename = string(dir,"/", base_name, "_$i.gif")
        @info "writing gif to $filename"
        Crux.gif(s.mdp, s.agent.π, filename, Neps=Neps)
    end
end

