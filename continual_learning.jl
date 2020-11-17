using Flux
using Plots

function continual_learning!(model, train_fn, train_data, test_data, eval_fn; optimizers = [ADAM(1e-3) for _=1:length(train_data)], epochs = 10)
    N = length(train_data)
    eval_curves = Dict(i => Float64[] for i=1:N)
    for i=1:N
        println("training on task $i")
        for e=1:epochs
            for j=1:i
                push!(eval_curves[j], eval_fn(model, test_data[j]))
            end
            train_fn(model, train_data[i], optimizers[i])
        end
    end 
    eval_curves   
end


function plot_results(eval_curves; task_names = ["Task $i" for i=1:length(eval_curves)], eval_metric = "Accuracy", title = "Training on Sequential Tasks")
    offset, epochs = 0, length(eval_curves[length(eval_curves)])
    p = plot(title = title, ylabel = eval_metric, xlabel="Training Steps", legend = :bottomright)
    for i=1:length(eval_curves)
        y = eval_curves[i]
        x = offset:offset + length(y) - 1
        plot!(p, x, y, label=task_names[i])
        offset += epochs
    end
    p    
end

