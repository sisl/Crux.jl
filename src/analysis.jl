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
function cumulative_rewards(input, key = :undiscounted_return)
    dirs = directories(input)
    x, y, breaks, max_iter = [], [], [], 0
    for dir in dirs
        hist = readtb(dir)[key]
        push!(y, hist.values...)
        push!(x, (hist.iterations .+ max_iter)...)
        max_iter = maximum(x)
        push!(breaks, max_iter)
    end
    x, cumsum(y), breaks
end

function plot_cumulative_rewards(input; p = plot(), label = "", key = :undiscounted_return)
    x, y, lines = cumulative_rewards(input, key)

    plot!(p, x, y, label = label, title = "Cumulative Reward", xlabel = "Test Index", ylabel = "Return")
    
    yrange = [extrema(y)...]
    for l in lines
        plot!(p, [l,l], yrange, color = :black, label = "")
    end
    p
end



function plot_jumpstart(input; p = plot(), key = :undiscounted_return, label = "")
    dirs = directories(input)
    jumpstarts = [readtb(dir, key)[2][1] for dir in dirs]
    plot!(p, jumpstarts, xlabel = "Task Iteration", marker = true, markersize = 3, ylabel = "Return", title = "Jumpstart Performance", label = label)
end

function plot_peak_performance(input; p = plot(), key = :undiscounted_return, label = "", smooth_weight = 0.6)
    dirs = directories(input)
    jumpstarts = [maximum(smooth(readtb(dir, key)[2], smooth_weight)) for dir in dirs]
    plot!(p, jumpstarts, xlabel = "Task Iteration", marker = true, markersize = 3, ylabel = "Return", title = "Peak Performance", label = label)
end

function plot_steps_to_threshold(input, thresh; p = plot(), key = :undiscounted_return, label = "", smooth_weight = 0.6)
    dirs = directories(input)
    steps = []
    for dir in dirs    
        iters, vals = readtb(dir, key)
        push!(steps, find_crossing(iters, smooth(vals, smooth_weight), thresh)[1])
    end
    plot!(p, steps, xlabel = "Task Iteration", marker = true, markersize = 3, ylabel = "Iterations", title = "Steps to $thresh performance", label = label)
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
    dirs = directories(input)
    
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


## Visualization
function episode_frames(mdp, policy; Neps = 1, max_steps = 1000, use_obs = false)
    frames = []
    for i = 1:Neps
        s = rand(initialstate(mdp))
        o = mdp isa POMDP ? rand(initialobs(mdp, s)) : convert_s(AbstractArray, s, mdp)
        step = 0
        while !isterminal(mdp, s) && step < max_steps
            step += 1
            a = action(policy, o)
            size(a) == (1,) && (a=a[1])
            if mdp isa POMDP
                s, o, _ = gen(mdp, s, a)
            else
                s, _ = gen(mdp, s, a)
                o = convert_s(AbstractArray, s, mdp)
            end
            use_obs ? push!(frames, o') : push!(frames, render(mdp, s, a))
        end
    end
    frames
end

gif(mdp, policy, filename; fps = 15, Neps = 1, max_steps = 1000, use_obs = false) = gif(episode_frames(mdp, policy, Neps = Neps, max_steps = max_steps, use_obs = use_obs), filename, fps = fps)

function gif(frames, filename; fps = 15)
    save(filename, cat(frames..., dims = 3), fps = fps)
end