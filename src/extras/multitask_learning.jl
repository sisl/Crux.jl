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
    push!(𝒮.log.extras, logfn([Sampler(t, 𝒮.π, 𝒮.S) for t in tasks]))
end

function continual_learning(tasks, solver_generator)
    solvers = Solver[]
    𝒮 = solver_generator(i = 1)
    for i in 1:length(tasks)
        solve(𝒮, tasks[i])
        push!(solvers, deepcopy(𝒮))
        if i < length(tasks)
            𝒮 = solver_generator(tasks = tasks[1:i], solvers=solvers, i = i+1)
        end
    end
    solvers
end

function sequential_learning(solve_tasks, eval_tasks, solver)
    samplers = [Sampler(t, solver.π, solver.S) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers))
    for t in solve_tasks
        solve(solver, t)
    end
end

function experience_replay(solve_tasks, eval_tasks, solver; experience_buffer, steps_per_task, sampler_π_explore = nothing)
    samplers = [Sampler(t, solver.π, solver.S, solver.A) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers))
    for t in solve_tasks
        length(experience_buffer) > 0 ? solve(solver, t, experience_buffer) : solve(solver, t)
        sampler = Sampler(t, RandomPolicy(t), solver.S, solver.A,  π_explore=sampler_π_explore)
        push!(experience_buffer, steps!(sampler, Nsteps = steps_per_task))
    end
end


function ewc(solve_tasks, eval_tasks, solver; λ_fisher = 1f0, fisher_batch_size = 50, fisher_buffer_size = 1000)
    # Setup the regularizer
    θ = Flux.params(solver.π)
    solver.regularizer = DiagonalFisherRegularizer(θ, λ_fisher)
    
    # Construct the thing to log
    samplers = [Sampler(t, solver.π, solver.S, solver.A) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers, Neps = solver.eval_eps))
    for t in solve_tasks
        solve(solver, t)
        
        loss = (𝒟) -> -mean(solver.π.Q(𝒟[:s]) .* 𝒟[:a])
        
        # update the regularizer
        update_fisher!(solver.regularizer, solver.buffer, loss, θ, fisher_batch_size; i=0)
    end
end

