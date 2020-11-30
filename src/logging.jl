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
        d = dict()
        for (k,v) in d
            log_value(p.logger, k, v, step = i)
            p.verbose && print(", ", k, ": ", v)
        end
    end
    p.verbose && println()
end

# Built-in functions for logging common training things
log_performance(tasks::AbstractVector, solution, name, fn, rng) = Dict("$(name)/T$i" => fn(tasks[i], solution, rng = rng) for i=1:length(tasks))
log_performance(task, solution, name, fn, rng) = Dict(name => fn(task, solution; rng = rng))

log_discounted_return(task, solution, rng) = () -> log_performance(task, solution, "discounted_return", discounted_return, rng)
log_undiscounted_return(task, solution, rng) = () -> log_performance(task, solution, "undiscounted_return", undiscounted_return, rng)
log_failure(task, solution, rng) = () -> log_performance(task, solution, "failure_rate", failure, rng)

log_val(val; name, suffix = "") = () -> Dict(string(name, suffix) => val)
log_loss(loss::T; name = "loss", suffix = "") where T <: Real = log_val(loss, name = name, suffix = suffix)
log_loss(losses::T; name = "loss", suffix = "") where T <: AbstractArray = log_val(mean(losses), name = name, suffix = suffix)
log_gradient(grad::T; name = "grad_norm", suffix = "") where T <: Real = log_val(norm(grad), name = name, suffix = suffix)
log_gradient(grads::T; name = "grad_norm", suffix = "") where T <: AbstractArray = log_val(mean(norm.(grads)), name = name, suffix = suffix)
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

function plot_learning_curves(dirs; 
        ylabel = "Discounted Reward",  
        xlabel = "Training Steps", 
        values = fill(:discounted_return, length(dirs)), 
        p = plot(), 
        rng = MersenneTwister(5), 
        colors = [get(colorschemes[:rainbow], rand(rng)) for i=1:length(values)],
        vertical_lines = [],
        vline_range = (0, 1), 
        thick_every = 1
    )
    # Plot the vertical lines (usually for multitask learning or to designate a point on the curve)
    for i = 1:length(vertical_lines)
        plot!(p, [vertical_lines[i], vertical_lines[i]], [vline_range...], color = :black, linewidth = i % thick_every == 0 ? 3 : 1, label = "")
    end
    
    # Plot the learning curves
    plot!(p, ylabel = ylabel, legend = :bottomright)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        plot!(p, x, y, alpha = 0.3, color = colors[i], label = "")
        plot!(p, x, smooth(y), color = colors[i], label = "Task $i", linewidth =2 )
    end
    p
end

