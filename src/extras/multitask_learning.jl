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

function log_multitask_performances!(𝒮, tasks, logfn = log_undiscounted_return)
    push!(𝒮.log.extras, logfn([Sampler(t, 𝒮.π, 𝒮.S, rng = 𝒮.rng) for t in tasks]))
end

function sequential_learning(solve_tasks, eval_tasks, solver)
    samplers = [Sampler(t, solver.π, solver.S, rng = solver.rng) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers))
    for t in solve_tasks
        solve(solver, t)
    end
end

function experience_replay(solve_tasks, eval_tasks, solver; experience_buffer, steps_per_task, sampler_exploration_policy = nothing)
    samplers = [Sampler(t, solver.π, solver.S, solver.A, rng = solver.rng) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers))
    for t in solve_tasks
        length(experience_buffer) > 0 ? solve(solver, t, experience_buffer) : solve(solver, t)
        sampler = Sampler(t, RandomPolicy(t), solver.S, solver.A, rng = solver.rng,  exploration_policy = sampler_exploration_policy)
        push!(experience_buffer, steps!(sampler, Nsteps = steps_per_task))
    end
end


function ewc(solve_tasks, eval_tasks, solver; λ_fisher = 1f0, fisher_batch_size = 50, fisher_buffer_size = 1000)
    # Setup the regularizer
    θ = Flux.params(solver.π)
    solver.regularizer = DiagonalFisherRegularizer(θ, λ_fisher)
    
    # Construct the thing to log
    samplers = [Sampler(t, solver.π, solver.S, solver.A, rng = solver.rng) for t in eval_tasks]
    push!(solver.log.extras, log_undiscounted_return(samplers, Neps = solver.eval_eps))
    for t in solve_tasks
        solve(solver, t)
        
        loss = (𝒟) -> -mean(solver.π.Q(𝒟[:s]) .* 𝒟[:a])
        
        # update the regularizer
        update_fisher!(solver.regularizer, solver.buffer, loss, θ, fisher_batch_size; i=0, rng = solver.rng)
    end
end

