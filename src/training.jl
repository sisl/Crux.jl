@with_kw mutable struct TrainingParams
    loss
    optimizer = ADAM(3f-4)
    regularizer = (π) -> 0
    batch_size = 128
    epochs = 80
    update_every = 1
    early_stopping = (info) -> false
    name = ""
    max_batches=Inf
end

Flux.Optimise.train!(π::N, loss::Function, p::TrainingParams; info = Dict()) where {N <: Policy} = train!(Flux.params(π), (;info) -> loss(info = info) + p.regularizer(π), p, info=info)
    
function Flux.Optimise.train!(θ, loss::Function, p::TrainingParams; info = Dict())
    l, back = Flux.pullback(() -> loss(info = info), θ)
    typeof(l) == Float64 && @error "Float64 loss found: computation in double precision may be slow"
    grad = back(1f0)
    gnorm = norm(grad, p=2)
    isnan(gnorm) && error("NaN detected! Loss: $l")
    Flux.update!(p.optimizer, θ, grad)
    info[string(p.name, "loss")] = l
    info[string(p.name, "grad_norm")] = gnorm
    info
end

# Train with minibatches and epochs
function batch_train!(π, p::TrainingParams, 𝒫, 𝒟::ExperienceBuffer...; info=Dict(), π_loss=π)
    infos = [] # stores the aggregated info for each epoch
    total_batches = 0
    for epoch in 1:p.epochs
        minibatch_infos = [] # stores the info from each minibatch
        
        # Shuffle the experience buffers
        for D in 𝒟
            shuffle!(D)
        end
        
        # Call train for each minibatch
        partitions = [partition(1:length(D), p.batch_size) for D in 𝒟]
        for indices in zip(partitions...)
            mbs = [minibatch(D, i) for (D, i) in zip(𝒟, indices)] 
            push!(minibatch_infos, train!(π, (;kwargs...)->p.loss(π_loss, 𝒫, mbs...; kwargs...), p, info=info))
            total_batches += 1 
            if total_batches >= p.max_batches
                break
            end
            
            if p.early_stopping([infos...,  aggregate_info(minibatch_infos)])
                info[:early_stop_batch] = total_batches
                break    
            end
        end
        push!(infos, aggregate_info(minibatch_infos))
        if p.early_stopping(infos)
            break    
        end
        
    end
    merge!(info, aggregate_info(infos))
end

