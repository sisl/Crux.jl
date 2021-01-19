elapsed(i::Int, N::Int) = (i % N) == 0
elapsed(i::UnitRange, N::Int) = any([i...] .% N .== 0)

@with_kw mutable struct LoggerParams
    dir::String = "log/"
    period::Int64 = 100
    logger = TBLogger(dir, tb_increment)
    extras = []
    verbose::Bool = true
end

Base.log(p::Nothing, i, data...)  = nothing

#Note that i can be an int or a unitrange
function Base.log(p::LoggerParams, i::Union{Int, UnitRange}, data...)
    !elapsed(i, p.period) && return
    i = i[end]
    p.verbose && print("Step: $i")
    for dict in [data..., p.extras...]
        d = dict isa Function ? dict() : dict
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
    res = Dict(k => mean([info[k] for info in infos]) for k in keys(infos[1]))
end

# Built-in functions for logging common training things
log_performance(s::AbstractVector, name, fn; kwargs...) = Dict("$(name)/T$i" => fn(s[i]; kwargs...) for i=1:length(s))
log_performance(s::Sampler, name, fn; kwargs...) = Dict(name => fn(s; kwargs...))

log_discounted_return(s; kwargs...) = () -> log_performance(s, "discounted_return", discounted_return; kwargs...)
log_undiscounted_return(s; kwargs...) = () -> log_performance(s, "undiscounted_return", undiscounted_return; kwargs...)
log_failure(s; kwargs...) = () -> log_performance(s, "failure_rate", failure; kwargs...)

log_val(val; name, suffix = "") = () -> Dict(string(name, suffix) => val)
log_val(val::T; name, suffix = "") where T <: AbstractArray = () -> Dict(string(name, suffix) => mean(val))
log_loss(loss; name = "loss", suffix = "") = log_val(loss, name = name, suffix = suffix)
log_gradient(grad; name = "grad_norm", suffix = "") = log_val(grad, name = name, suffix = suffix)
log_exploration(policy, i; name = "eps") = () -> policy isa EpsGreedyPolicy ? Dict(name => policy.eps(i)) : Dict()

## Stuff for plotting
function smooth(v, weight = 0.6)
    N = length(v)
    smoothed = Array{Float64, 1}(undef, N)
    smoothed[1] = v[1]
    for i = 2:N
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * v[i]
    end
    smoothed
end

function readtb(logdir)
    hist = MVHistory()
    TensorBoardLogger.map_summaries(logdir) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    hist
end

function readtb(logdir, key)
    h = readtb(logdir)[key]
    h.iterations, h.values
end

function plot_learning(input; 
        title = "",
        ylabel = "Undiscounted Return",  
        xlabel = "Training Steps", 
        values = :undiscounted_return, 
        labels = :default,
        legend = :bottomright,
        font = :palatino,
        p = plot(), 
        colors = (i) -> get(colorschemes[:rainbow], rand(MersenneTwister(5),i)[end]),
        vertical_lines = [],
        vline_range = (0, 1), 
        thick_every = 1
    )
    # Get the directories we care about
    dirs = input
    input isa Array && all([input[i] isa Solver for i=1:length(input)]) && (dirs = [input[i].log.logger.logdir for i=1:length(input)])
    input isa Solver && (dirs = [input.log.logger.logdir])
    input isa String && (dirs = [input])
    
    N = length(dirs)
    values isa Symbol && (values = fill(values, N))
    if labels == :default
        labels = N == 1 ? [""] : ["Task $i" for i=1:N]
    end 
    
    # Plot the vertical lines (usually for multitask learning or to designate a point on the curve)
    for i = 1:length(vertical_lines)
        plot!(p, [vertical_lines[i], vertical_lines[i]], [vline_range...], color = :black, linewidth = i % thick_every == 0 ? 3 : 1, label = "")
    end
    
    # Plot the learning curves
    plot!(p, ylabel = ylabel, xlabel = xlabel, legend = legend, title = title, fontfamily = font)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        plot!(p, x, y, alpha = 0.3, color = colors(i), label = "")
        plot!(p, x, smooth(y), color = colors(i), label = labels[i], linewidth =2 )
    end
    p
end

function episode_frames(mdp, policy, rng::AbstractRNG = Random.GLOBAL_RNG; Neps = 1, max_steps = 1000, use_obs = false)
    frames = []
    for i = 1:Neps
        s = rand(initialstate(mdp))
        o = mdp isa POMDP ? rand(initialobs(mdp, s)) : convert_s(AbstractArray, s, mdp)
        step = 0
        while !isterminal(mdp, s) && step < max_steps
            step += 1
            a = action(policy, o)
            if mdp isa POMDP
                s, o, _ = gen(mdp, s, a, rng)
            else
                s, _ = gen(mdp, s, a, rng)
                o = convert_s(AbstractArray, s, mdp)
            end
            use_obs ? push!(frames, o') : push!(frames, render(mdp, s, a))
        end
    end
    frames
end

gif(mdp, policy, filename; rng::AbstractRNG = Random.GLOBAL_RNG, fps = 15, Neps = 1, max_steps = 1000, use_obs = false) = gif(episode_frames(mdp, policy, rng, Neps = Neps, max_steps = max_steps, use_obs = use_obs), filename, fps = fps)

function gif(frames, filename; fps = 15)
    save(filename, cat(frames..., dims = 3), fps = fps)
end