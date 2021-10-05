POMDPs.discount(v::AbstractVector) = discount(v[1])



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
