include("../src/includes.jl")
include("mdps/lavaworld.jl")
using Plots

## Continual Learning params
Ncycles = 3
Ntasks = 3
Nsteps_per_cycle = 15000
N = Ncycles*Ntasks*Nsteps_per_cycle

## Build the tasks
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
rng = MersenneTwister(2)
tasks = [SimpleGridWorld(size = sz, tprob = 0.99, rewards = random_lava(sz, 6, rng = rng)) for _=1:Ntasks]
as = actions(tasks[1])
render_and_save("lavaworld_tasks.pdf", tasks...)

## Define the network we are using
Q(args...) = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4, sigmoid), args...)

## Train individually
solve(DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = N, log = LoggerParams(dir = "log/ind_task1")), tasks[1])
solve(DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = N, log = LoggerParams(dir = "log/ind_task2")), tasks[2])
solve(DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = N, log = LoggerParams(dir = "log/ind_task3")), tasks[3])

plot_learning_curves(["log/ind_task1/", "log/ind_task2/", "log/ind_task3/"])
savefig("trained_separately.pdf")


## Train Jointly
𝒮_joint = DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = N, batch_size = 96, log = LoggerParams(dir = "log/joint"))
solve(𝒮_joint, tasks)

plot_learning_curves(fill("log/joint/", 3), values = [Symbol("discounted_return/T1"), Symbol("discounted_return/T2"), Symbol("discounted_return/T3") ])
savefig("trained_jointly.pdf")


## Train Sequentially
seq_tasks = repeat(tasks, Ncycles)
𝒮_seq = DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = Nsteps_per_cycle, 
                  exploration_policy = EpsGreedyPolicy(MultitaskDecaySchedule(Nsteps_per_cycle, 1:length(seq_tasks)), rng, as),
                  log = LoggerParams(dir = "log/continual"))
sequential_learning(seq_tasks, tasks, 𝒮_seq)

p = plot_learning_curves(fill("log/continual/", 3), values = [Symbol("discounted_return/T1"), Symbol("discounted_return/T2"), Symbol("discounted_return/T3")], vertical_lines = [i*Nsteps_per_cycle for i=1:length(seq_tasks)], thick_every = 3, vline_range = (-0.5, 0.85))
savefig("trained_sequentially.pdf")

## Continual Learning with experience Replay
𝒮_er = DQNSolver(π = DQNPolicy(Q(), as), sdim = input_dim, N = Nsteps_per_cycle, 
                  exploration_policy = EpsGreedyPolicy(MultitaskDecaySchedule(Nsteps_per_cycle, 1:length(seq_tasks)), rng, as),
                  log = LoggerParams(dir = "log/er"))
experience_replay(seq_tasks, tasks, 𝒮_er, experience_buffer = ExperienceBuffer(𝒮_er.sdim, 𝒮_er.adim, 2000), steps_per_task = 1000)

p = plot_learning_curves(fill("log/er/", 3), values = [Symbol("discounted_return/T1"), Symbol("discounted_return/T2"), Symbol("discounted_return/T3")], vertical_lines = [i*Nsteps_per_cycle for i=1:length(seq_tasks)], thick_every = 3, vline_range = (-0.5, 0.85))
savefig("trained_sequentially_with_replay.pdf")


