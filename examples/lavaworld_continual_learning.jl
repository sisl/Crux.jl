include("../src/includes.jl")
include("mdps/lavaworld.jl")

## Setup the continual learning cycle
function continual_rl!(tasks, solver; cycles = 10)
    N = length(tasks)
    for i=1:cycles
        println("=============== cycle $i =================")
        for t=1:N
            println("================ task $t ==================")
            solver.π.mdp = tasks[t]
            solve(solver, tasks[t])
        end
    end 
end

## Continual Learning params
Ncycles = 10
Ntasks = 2
Nsteps_per_cycle = 10000

## Build the tasks
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
tasks = [SimpleGridWorld(size = sz, tprob = 1.0, rewards = random_lava(sz, 6)) for _=1:Ntasks]
simple_display(tasks[1])
simple_display(tasks[2])
# simple_display(tasks[3])


## Build the policy
Q = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4))
p = DQNPolicy(Q, tasks[2])


## Build the solver
𝒮 = DQNSolver(π = p, 
              N=Nsteps_per_cycle, 
              batch_size = 128, 
              exploration_policy = EpsGreedyPolicy(tasks[2], LinearDecaySchedule(start=1., stop=0.1, steps=Ncycles*Ntasks*Nsteps_per_cycle/2)),
              log = LoggerParams(dir = "log/continual_lavaworld", period = 300, eval = (a, b; kwargs...) -> Dict(), other = (p) -> Dict("perf/T$i" => discounted_return(tasks[i], DQNPolicy(p.Q, tasks[i])) for i=1:Ntasks))
              )

## Run
continual_rl!(tasks, 𝒮, cycles = Ncycles)

