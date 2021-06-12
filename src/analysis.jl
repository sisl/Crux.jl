## Reading Logs
function readtb(logdir::String)
    hist = MVHistory()
    TensorBoardLogger.map_summaries(logdir) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    hist
end

function readtb(logdir::String, key)
    h = readtb(logdir)[key]
    h.iterations, h.values
end

# Convert various inputs to an arrow of directories for plotting
directories(input::String) = [input]
directories(input::AbstractArray{Any}) = [i isa Solver ? i.log.logger.logdir : i for i in input]
directories(input::AbstractArray{T}) where {T <: String} = input
directories(input::AbstractArray{T}) where {T <: Solver} = [input[i].log.logger.logdir for i=1:length(input)]

## Learning Curve Metrics
# compute a percentile between a known range
percentile(frac, min, max) = frac*(max - min) + min

# Find the point in x that y exceeds val
function find_crossing(x, y, val)
    i = findfirst(y .>= val)
    isnothing(i) ? NaN : x[i]
end

# Exponential Smoothing
function smooth(v, weight = 0.6)
    N = length(v)
    smoothed = Array{Float64, 1}(undef, N)
    smoothed[1] = v[1]
    for i = 2:N
        smoothed[i] = smoothed[i-1] * weight + (1 - weight) * v[i]
    end
    smoothed
end

## Continual Learning Analysis
# Compiles the cumulative reward while learning a sequence of tasks
function cumulative_rewards(input, key=(i) -> Symbol("undiscounted_return/T$i"))
    dirs = directories(input)
    x, y, breaks, max_iter = [], [], [], 0
    for (dir, i) in zip(dirs, 1:length(dirs))
        hist = readtb(dir)[key(i)]
        push!(y, hist.values...)
        push!(x, (hist.iterations .+ max_iter)...)
        max_iter = maximum(x)
        push!(breaks, max_iter)
    end
    x, cumsum(y), breaks
end

# Compiles the performance on each task over time
function single_task_performances(input, key=(i) -> Symbol("undiscounted_return/T$i"))
    dirs = directories(input)
    res = Dict()
    breaks = []
    offset = 0
    for t=1:length(dirs) # Loop over each task
        for i=1:t # Loop up to each task
            try
            x, y = readtb(dirs[t], key(i)) # read the data
            if i==t 
                res[i] = ([], []) # Empty data for task t 
                push!(breaks, maximum(x))
            end           

            push!(res[i][1], offset .+ x ...)
            push!(res[i][2], y...)
            catch end
        end
        offset = maximum(res[t][1])
    end
    res, cumsum(breaks)
end

function plot_cumulative_rewards(input; p=plot(), key=(i)->Symbol("undiscounted_return/T$i"), show_lines=false, kwargs...)
    x, y, lines = cumulative_rewards(input, key)

    plot!(p, x, y, title="Cumulative Reward", xlabel="Training Iteration", ylabel="Return"; kwargs...)
    
    if show_lines 
        yrange = [0.9, 1.5].*[extrema(y)...]
        for l in lines
            plot!(p, [l,l], yrange, color=:black, label = "")
        end
    end
    p
end

function plot_jumpstart(input; p=plot(), key=(i)->Symbol("undiscounted_return/T$i"), kwargs...)
    dirs = directories(input)
    jumpstarts = [readtb(dir, key(i))[2][1] for (dir, i) in zip(dirs,1:length(dirs))]
    plot!(p, jumpstarts, xlabel = "Task Iteration", marker=true, markersize=3, ylabel="Return", title="Jumpstart Performance"; kwargs...)
end

function plot_peak_performance(input; p=plot(), key=(i)->Symbol("undiscounted_return/T$i"), smooth_weight=0.6, kwargs...)
    dirs = directories(input)
    jumpstarts = [maximum(smooth(readtb(dir, key(i))[2], smooth_weight)) for (dir, i) in zip(dirs,1:length(dirs))]
    plot!(p, jumpstarts, xlabel = "Task Iteration", marker=true, markersize=3, ylabel="Return", title="Peak Performance"; kwargs...)
end

function plot_steps_to_threshold(input, thresh; p=plot(), key=(i)->Symbol("undiscounted_return/T$i"), smooth_weight=0.6, kwargs...)
    dirs = directories(input)
    steps = []
    for (dir,i) in zip(dirs, 1:length(dirs))    
        iters, vals = readtb(dir, key(i))
        push!(steps, find_crossing(iters, smooth(vals, smooth_weight), thresh)[1])
    end
    plot!(p, steps, xlabel = "Task Iteration", marker = true, markersize = 3, ylabel = "Iterations", title = "Steps to $thresh performance"; kwargs...)
end


function plot_forgetting(input; p=plot([plot() for _=1:length(input)]..., layout=(length(input),1)), key=(i)->Symbol("undiscounted_return/T$i"), smooth_weight=0.6, kwargs...)
    curves, lines = single_task_performances(input)
    for (k,v) in curves
        plot!(p[k], v[1], smooth(v[2], smooth_weight), xlabel="Training Iteration", ylabel="Return", xlims=(0,Inf); kwargs...)
    end
    p
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
        vertical_lines = [],
        vline_range = (0, 1), 
        thick_every = 1
    )
    dirs = directories(input)
    
    N = length(dirs)
    values isa Symbol && (values = fill(values, N))
    if labels == :default
        labels = N == 1 ? [""] : ["Task $i" for i=1:N]
    end 
    
    # Plot the vertical lines (usually for multitask learning or to designate a point on the curve)
    for i = 1:length(vertical_lines)
        plot!(p, [vertical_lines[i], vertical_lines[i]], [vline_range...], color=:black, linewidth = i % thick_every == 0 ? 3 : 1, label = "")
    end
    
    # Plot the learning curves
    plot!(p, ylabel = ylabel, xlabel = xlabel, legend = legend, title = title, fontfamily = font)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        plot!(p, x, y, alpha = 0.3, color=i, label = "")
        plot!(p, x, smooth(y), color=i, label = labels[i], linewidth =2 )
    end
    p
end


## Visualization
function episode_frames(mdp, policy; Neps=1, max_steps=1000, use_obs=false, render_kwargs::NamedTuple=(;), S=state_space(mdp))
    frames = []
    for i = 1:Neps
        s = rand(initialstate(mdp))
        o = mdp isa POMDP ? rand(initialobs(mdp, s)) : convert_s(AbstractArray, s, mdp)
        step = 0
        while !isterminal(mdp, s) && step < max_steps
            step += 1
            a = action(policy, tovec(o,S))
            size(a) == (1,) && (a=a[1])
            if mdp isa POMDP
                s, o, _ = gen(mdp, s, a)
            else
                s, _ = gen(mdp, s, a)
                o = convert_s(AbstractArray, s, mdp)
            end
            use_obs ? push!(frames, o') : push!(frames, render(mdp, s, a; render_kwargs...))
        end
    end
    frames
end

function gif(mdp, data::ExperienceBuffer, filename; convert_s = (s)->s, convert_a=(a)->a, render_kwargs::NamedTuple=(;), kwargs...)
    frames = [render(mdp, convert_s(bslice(data[:s], i)), convert_a(bslice(data[:a], i)); render_kwargs...) for i=1:length(data)]
    gif(frames, filename; kwargs...)
end

gif(mdp, policy, filename; fps=15, Neps=1, max_steps=1000, use_obs=false, S=state_space(mdp), render_kwargs::NamedTuple=(;)) = gif(episode_frames(mdp, policy, Neps=Neps, max_steps=max_steps, use_obs=use_obs, render_kwargs=render_kwargs, S=S), filename, fps=fps)

function gif(frames, filename; fps=15)
    save(filename, cat(frames..., dims=3), fps=fps)
end

