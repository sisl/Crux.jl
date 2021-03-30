POMDPs.discount(v::AbstractVector) = discount(v[1])

function MultitaskDecaySchedule(steps::Int, task_ids; start = 1.0, stop = 0.1)
    schedule = LinearDecaySchedule(start, stop, steps)
    function val(i)
        taskindex = ceil(Int, i / steps)
        taskindex < 1 && return start
        taskindex > length(task_ids) && return stop
        taskid = task_ids[taskindex]
        used_steps = steps*sum(task_ids[1:taskindex-1] .== taskid)
        schedule(used_steps + mod1(i, steps))
    end
end

function log_multitask_performances!(𝒮, tasks, logfn=log_undiscounted_return)
    push!(𝒮.log.extras, logfn([Sampler(t, 𝒮.π) for t in tasks]))
end

function continual_learning(tasks, solver_generator)
    solvers = Solver[]
    history = []
    𝒮 = solver_generator(i=1, tasks=tasks[1:1], history=history)
    for i in 1:length(tasks)
        solve(𝒮, tasks[i])
        push!(solvers, deepcopy(𝒮))
        if i < length(tasks)
            𝒮 = solver_generator(tasks=tasks[1:i+1], solvers=solvers, i=i+1, history=history)
        end
    end
    solvers
end
