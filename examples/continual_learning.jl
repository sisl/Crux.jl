using Crux, Flux, POMDPPolicies, POMDPs, POMDPGym, Random, Plots


## Build the tasks
N = 20000
Ntasks = 3
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
rng = MersenneTwister(2)
tasks = [LavaWorldMDP(size = sz, tprob = 0.99, goal = :random, randomize_lava = false, rng = rng, num_lava_tiles = 6) for _=1:Ntasks]
S = state_space(tasks[1])
as = [actions(tasks[1])...]
render(tasks[1])

# Define the network
Q() = DiscreteNetwork(Chain(x->reshape(x, input_dim, :), Dense(input_dim, 64, relu), Dense(64,64, relu), Dense(64, 4)), as)

## from scratch
from_scratch(;i, kwargs...) = DQNSolver(π = Q(), S = S, N = N, log = LoggerParams(dir = "log/task$i"))

## warm start
function warm_start(;i, solvers = [], tasks = []) 
    # Copy over the previous policy 
    pol = isempty(solvers) ? Q() : deepcopy(solvers[end].π)
    
    # Construct the solver
    𝒮 = DQNSolver(π = pol, S = S, N = N, log = LoggerParams(dir = "log/task$i"))
    
    # Record performance on previous tasks
    i > 1 && push!(𝒮.log.extras, log_undiscounted_return([Sampler(t, 𝒮.π, 𝒮.S, rng = 𝒮.rng) for t in tasks[1:i-1]]))
    𝒮
end

scratch_solvers = continual_learning(tasks, from_scratch)
warmstart_solvers = continual_learning(tasks, warm_start)

using BSON, TensorBoardLogger, StaticArrays, POMDPModels
BSON.@save "scratch_solvers.bson" scratch_solvers
BSON.@save "warmstart_solvers.bson" warmstart_solvers

scratch_solvers = BSON.load("scratch_solvers.bson")[:scratch_solvers]
warmstart_solvers = BSON.load("warmstart_solvers.bson")[:warmstart_solvers]

# cumulative_rewards
p_rew = Crux.plot_cumulative_rewards(scratch_solvers, label="scratch")
Crux.plot_cumulative_rewards(warmstart_solvers, p=p_rew, label="warm start")

p_jump = plot_jumpstart(scratch_solvers, label="scratch")
plot_jumpstart(warmstart_solvers, p=p_jump, label="warm start")

p_perf = plot_peak_performance(scratch_solvers, label="scratch")
plot_peak_performance(warmstart_solvers, p=p_perf, label="warm start")

p_thresh = Crux.plot_steps_to_threshold(scratch_solvers, .99, label="scratch")
Crux.plot_steps_to_threshold(warmstart_solvers, .99, p=p_thresh, label="warm start")


# plot_forgetting(warmstart_solvers)



## Continual Learning params
Ncycles = 2
Ntasks = 3
Nsteps_per_cycle = 10000
N = Ncycles*Ntasks*Nsteps_per_cycle

## Build the tasks
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
rng = MersenneTwister(2)
tasks = [LavaWorldMDP(size = sz, tprob = 0.99, goal = :random, randomize_lava = false, rng = rng, num_lava_tiles = 6) for _=1:Ntasks]
S = state_space(tasks[1])
as = [actions(tasks[1])...]
render(tasks[1])
# render_and_save("lavaworld_tasks.pdf", tasks...)

## Define the network we are using
Q() = DiscreteNetwork(Chain(x->reshape(x, input_dim, :), Dense(input_dim, 64, relu), Dense(64,64, relu), Dense(64, 4)), as)

## Train individually
solve(DQNSolver(π = Q(), S = S, N = N, log = LoggerParams(dir = "log/ind_task1")), tasks[1])
solve(DQNSolver(π = Q(), S = S, N = N, log = LoggerParams(dir = "log/ind_task2")), tasks[2])
solve(DQNSolver(π = Q(), S = S, N = N, log = LoggerParams(dir = "log/ind_task3")), tasks[3])

plot_learning(["log/ind_task1/", "log/ind_task2/", "log/ind_task3/"], title="LavaWorld Training - 3 Tasks")
savefig("trained_separately.pdf")


## Train Jointly
𝒮_joint = DQNSolver(π = Q(), S = S, N = N, batch_size = 96, log = LoggerParams(dir = "log/joint"))
solve(𝒮_joint, tasks)


plot_learning(fill("log/joint/", 3), values = [Symbol("undiscounted_return/T1"), Symbol("undiscounted_return/T2"), Symbol("undiscounted_return/T3") ])
savefig("trained_jointly.pdf")


## Train Sequentially
seq_tasks = repeat(tasks, Ncycles)
𝒮_seq = DQNSolver(π = Q(), S = S, N = Nsteps_per_cycle, 
                  π_explore = ϵGreedyPolicy(MultitaskDecaySchedule(Nsteps_per_cycle, 1:length(seq_tasks)), rng, as),
                  log = LoggerParams(dir = "log/continual"))
sequential_learning(seq_tasks, tasks, 𝒮_seq)

p = plot_learning(fill(𝒮_seq, 3), values = [Symbol("undiscounted_return/T1"), Symbol("undiscounted_return/T2"), Symbol("undiscounted_return/T3")], vertical_lines = [i*Nsteps_per_cycle for i=1:length(seq_tasks)], thick_every = 3, vline_range = (-0.5, 0.85))
savefig("trained_sequentially.pdf")

## Continual Learning with experience Replay
𝒮_er = DQNSolver(π = Q(), S = S, N = Nsteps_per_cycle, 
                  π_explore = ϵGreedyPolicy(MultitaskDecaySchedule(Nsteps_per_cycle, 1:length(seq_tasks)), rng, as),
                  log = LoggerParams(dir = "log/er"))
experience_replay(seq_tasks, tasks, 𝒮_er, experience_buffer = ExperienceBuffer(𝒮_er.S, 𝒮_er.A, 2000), steps_per_task = 1000)

p = plot_learning(fill(𝒮_er, 3), values = [Symbol("undiscounted_return/T1"), Symbol("undiscounted_return/T2"), Symbol("undiscounted_return/T3")], vertical_lines = [i*Nsteps_per_cycle for i=1:length(seq_tasks)], thick_every = 3, vline_range = (-0.5, 0.85))
savefig("trained_sequentially_with_replay.pdf")


## Continual Learning with elastic weight consolidation 
𝒮_ewc = DQNSolver(π = Q(), S = S, N = Nsteps_per_cycle, 
                  π_explore = ϵGreedyPolicy(MultitaskDecaySchedule(Nsteps_per_cycle, 1:length(seq_tasks)), rng, as),
                  log = LoggerParams(dir = "log/ewc"))
ewc(seq_tasks, tasks, 𝒮_ewc, λ_fisher = 1f11, fisher_batch_size = 128)

p = plot_learning(fill(𝒮_ewc, 3), values = [Symbol("undiscounted_return/T1"), Symbol("undiscounted_return/T2"), Symbol("undiscounted_return/T3")], vertical_lines = [i*Nsteps_per_cycle for i=1:length(seq_tasks)], thick_every = 3, vline_range = (-0.5, 0.85))
savefig("trained_sequentially_with_ewc.pdf")


