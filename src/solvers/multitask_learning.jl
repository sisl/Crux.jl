POMDPs.discount(v::AbstractVector) = discount(v[1])

function MultitaskDecaySchedule(steps::Int, task_ids; start = 1.0, stop = 0.1)
    schedule = LinearDecaySchedule(start = start, stop = stop, steps = steps)
    function val(i)
        taskindex = ceil(Int, i / steps)
        taskindex < 1 && return start
        taskindex > length(task_ids) && return stop
        taskid = task_ids[taskindex]
        used_steps = steps*sum(task_ids[1:taskindex-1] .== taskid)
        schedule(used_steps + mod1(i, steps))
    end
end

function sequential_learning(solve_tasks, eval_tasks, solver)
    push!(solver.log.extras, log_discounted_return(eval_tasks, solver.π, solver.rng))
    for t in solve_tasks
        solve(solver, t)
    end
end

function experience_replay(solve_tasks, eval_tasks, solver; experience_buffer, steps_per_task, sampler_exploration_policy = nothing)
    push!(solver.log.extras, log_discounted_return(eval_tasks, solver.π, solver.rng))
    for t in solve_tasks
        solve(solver, t, experience_buffer)
        sampler = Sampler(mdp = t, π = RandomPolicy(t), exploration_policy = sampler_exploration_policy)
        push!(experience_buffer, steps!(sampler, Nsteps = steps_per_task))
    end
end

