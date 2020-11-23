include("../src/includes.jl")
include("mdps/lavaworld.jl")
using Plots

## Setup the continual learning cycle
function continual_rl!(tasks, solver; cycles = 10)
    N = length(tasks)
    for i=1:cycles
        println("=============== cycle $i =================")
        for t=1:N
            println("================ task $t ==================")
            solver.π.mdp = tasks[t]
            solve(solver, tasks[t], explore_offset = ((i-1)*N + (t-1))*solver.N)
        end
    end 
end

function experience_replay!(tasks, solver; cycles = 10)
    ER = ExperienceBuffer(tasks[1], solver.buffer.size*2)
    N = length(tasks)
    for i=1:cycles
        println("=============== cycle $i =================")
        for t=1:N
            println("================ task $t ==================")
            solver.π.mdp = tasks[t]
            solve(solver, tasks[t], explore_offset = ((i-1)*N + (t-1))*solver.N, extra_buffer = ER)
            push_episodes!(ER, solver.π.mdp, solver.buffer.size)
        end
    end 
end

## Continual Learning params
Ncycles = 3
Ntasks = 3
Nsteps_per_cycle = 15000
N = Ncycles*Ntasks*Nsteps_per_cycle
rng = MersenneTwister(5)
colors = [get(colorschemes[:rainbow], rand(rng)) for i=1:Ntasks]

## Build the tasks
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
rng = MersenneTwister(2)
tasks = [SimpleGridWorld(size = sz, tprob = 0.99, rewards = random_lava(sz, 6, rng = rng)) for _=1:Ntasks]
render_and_save("lavaworld_tasks.pdf", tasks...)


## Define the network we are using
Q(args...) = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4, sigmoid), args...)

## Train individually
function train_task(tasks, i)
    𝒮 = DQNSolver(π = DQNPolicy(Q(), tasks[i]), 
                  N=N, 
                  exploration_policy = EpsGreedyPolicy(tasks[i], LinearDecaySchedule(start=1., stop=0.1, steps=N/2)),
                  log = LoggerParams(dir = "log/ind_task$i", period = 100),
                  opt = ADAM(1f-4)
                  )
    solve(𝒮, tasks[i])
end
train_task(tasks, 1)
train_task(tasks, 2)
train_task(tasks, 3)

# Make some plots
function plot_learning_curves(dirs; values = fill(:discounted_return, length(dirs)), p = plot())
    plot!(p, xlabel = "Training Steps", ylabel = "Discounted Reward", legend = :bottomright)
    for i in 1:length(dirs)
        x, y = readtb(dirs[i], values[i])
        plot!(p, x, y, alpha = 0.3, color = colors[i], label = "")
        plot!(p, x, smooth(y), color = colors[i], label = "Task $i", linewidth =2 )
    end
    p
end

plot_learning_curves(["log/ind_task1/", "log/ind_task2/", "log/ind_task3/"])
savefig("trained_separately.pdf")


## Train Jointly
Qnet = Q()
p = DQNPolicy(Qnet, tasks[1])
𝒮 = DQNSolver(π = p, 
              N=N, 
              batch_size = 96,
              exploration_policy = EpsGreedyPolicy(tasks[1], LinearDecaySchedule(start=1., stop=0.1, steps=N/2)),
              opt = ADAM(1f-4),
              buffer = BufferParams(init = 600, size = 3000),
              log = LoggerParams(dir = "log/simul/", period = 100, eval = (a, b; kwargs...) -> Dict(), other = (p) -> Dict("perf/T$i" => discounted_return(tasks[i], DQNPolicy(p.Q, tasks[i])) for i=1:Ntasks))
              )
init = ceil(Int, 𝒮.buffer.init / 3. )
buffers = [fill(ExperienceBuffer, t, init) for t in tasks]
buffer = merge(buffers..., capacity = 𝒮.buffer.size)
solve_multiple(𝒮, tasks..., buffer = buffer)

plot_learning_curves(fill("log/simul/", 3), values = [Symbol("perf/T1"), Symbol("perf/T2"), Symbol("perf/T3") ])
savefig("trained_simultaneously.pdf")


## Train Sequentially
Qnet = Q()
p = DQNPolicy(Qnet, tasks[1])
𝒮 = DQNSolver(π = p, 
              N=Nsteps_per_cycle, 
              batch_size = 32, 
              opt = ADAM(1f-4),
              exploration_policy = EpsGreedyPolicy(tasks[2], LinearDecaySchedule(start=1., stop=0.1, steps=Nsteps_per_cycle)),
              log = LoggerParams(dir = "log/seq/", period = 100, eval = (a, b; kwargs...) -> Dict(), other = (p) -> Dict("perf/T$i" => discounted_return(tasks[i], DQNPolicy(p.Q, tasks[i])) for i=1:Ntasks))
              )
continual_rl!(tasks, 𝒮, cycles = Ncycles)



xD, yD = Dict(), Dict()
TensorBoardLogger.map_summaries("log/seq/") do tag, iter, val
    s = Symbol(tag)
    if haskey(xD, s)
        push!(xD[s], iter)
        push!(yD[s], val)
    else
        xD[s] = [iter]
        yD[s] = [val]
    end
end

p = plot()
for i = 1:Ntasks*Ncycles
    plot!(p, [i*Nsteps_per_cycle, i*Nsteps_per_cycle], [-0.5, 0.85], color = :black, linewidth = i % 3 == 0 ? 3 : 1, label = "")
end
p = plot_learning_curves(fill("log/seq/", 3), p = p, values = [Symbol("perf/T1"), Symbol("perf/T2"), Symbol("perf/T3") ])

vs = [Symbol("perf/T1"), Symbol("perf/T2"), Symbol("perf/T3")]
plot!(p, xlabel = "Training Steps", ylabel = "Discounted Reward", legend = :bottomright)
for i in 1:length(vs)
    x, y = xD[vs[i]], yD[vs[i]]
    plot!(p, x, y, alpha = 0.3, color = colors[i], label = "")
    plot!(p, x, smooth(y), alpha = 0.7, color = colors[i], label = "Task $i", linewidth =2 )
end

p
savefig("trained_sequentially.pdf")

## Continual Learning with experience Replay
Qnet = Q()
p = DQNPolicy(Qnet, tasks[1])
𝒮 = DQNSolver(π = p, 
              N=Nsteps_per_cycle, 
              batch_size = 64, 
              opt = ADAM(1f-4),
              exploration_policy = EpsGreedyPolicy(tasks[2], LinearDecaySchedule(start=1., stop=0.1, steps=Nsteps_per_cycle)),
              log = LoggerParams(dir = "log/seq_er/", period = 100, eval = (a, b; kwargs...) -> Dict(), other = (p) -> Dict("perf/T$i" => discounted_return(tasks[i], DQNPolicy(p.Q, tasks[i])) for i=1:Ntasks))
              )
experience_replay!(tasks, 𝒮, cycles = Ncycles)

xD, yD = Dict(), Dict()
TensorBoardLogger.map_summaries("log/seq_er/") do tag, iter, val
    s = Symbol(tag)
    if haskey(xD, s)
        push!(xD[s], iter)
        push!(yD[s], val)
    else
        xD[s] = [iter]
        yD[s] = [val]
    end
end

p = plot()
for i = 1:Ntasks*Ncycles
    plot!(p, [i*Nsteps_per_cycle, i*Nsteps_per_cycle], [-0.5, 0.85], color = :black, linewidth = i % 3 == 0 ? 3 : 1, label = "")
end

vs = [Symbol("perf/T1"), Symbol("perf/T2"), Symbol("perf/T3")]
plot!(p, xlabel = "Training Steps", ylabel = "Discounted Reward", legend = :bottomright)
for i in 1:length(vs)
    x, y = xD[vs[i]], yD[vs[i]]
    plot!(p, x, y, alpha = 0.3, color = colors[i], label = "")
    plot!(p, x, smooth(y), alpha = 0.7, color = colors[i], label = "Task $i", linewidth =2 )
end

p
savefig("experience_replay.pdf")

