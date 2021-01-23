function Flux.Optimise.train!(π, loss::Function, opt; 
        regularizer = (π) -> 0, 
        loss_sym = :loss, 
        grad_sym = :grad_norm,
        info = Dict())
    θ = Flux.params(π)
    l, back = Flux.pullback(() -> loss(info = info) + regularizer(π), θ)
    typeof(l) == Float64 && @warn "Float64 loss found: computation in double precision may be slow"
    grad = back(1f0)
    gnorm = norm(grad, p=2)
    @assert !isnan(gnorm)
    Flux.update!(opt, θ, grad)
    info[loss_sym] = l
    info[grad_sym] = gnorm
    info
end

# Train with minibatches and epochs
function Flux.Optimise.train!(π, loss::Function, batch_size::Int, opt, 𝒟::ExperienceBuffer...; 
        epochs = 1, 
        regularizer = (π) -> 0, 
        early_stopping = (info) -> false,
        loss_sym = :loss, 
        grad_sym = :grad_norm,
        rng::AbstractRNG = Random.GLOBAL_RNG,
        )
    infos = [] # stores the aggregated info for each epoch
    for epoch in 1:epochs
        minibatch_infos = [] # stores the info from each minibatch
        
        # Shuffle the experience buffers
        for D in 𝒟
            shuffle!(rng, D)
        end
        
        # Call train for each minibatch
        partitions = [partition(1:length(D), batch_size) for D in 𝒟]
        for indices in zip(partitions...)
            mbs = [minibatch(D, i) for (D, i) in zip(𝒟, indices)] 
            push!(minibatch_infos, train!(π, (;kwargs...)->loss(π, mbs...; kwargs...), opt, regularizer = regularizer, loss_sym = loss_sym, grad_sym = grad_sym))
        end
        push!(infos, aggregate_info(minibatch_infos))
        if early_stopping(infos[end])
            println("early stopping at epoch $epoch")
            break    
        end
    end
    aggregate_info(infos)
end