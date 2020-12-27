function init_fisher_diagonal(params)
    F = [zeros(size(p)) for p in params]
    F, 0
end

function add_fisher_information_diagonal!(F, neg_loss, θ, N)
    # Compute the gradient of the negative loss
    grads = gradient(neg_loss, θ)
    
    # The diagonal entries are the square of the gradients (keep a running average)
    for (p, i) in zip(θ, 1:length(θ))
        F[i] += (grads[p].^2 .- F[i]) ./ N
    end    
end

function update_fisher_diagonal!(F, N, buffer, loss, θ, Nbatches, batch_size; i=0, rng::AbstractRNG = Random.GLOBAL_RNG)
    # Construct the minibatch buffer
    𝒟 = ExperienceBuffer(dim(buffer, :s), dim(buffer, :a), batch_size, device = device(buffer))
    for i=1:Nbatches
        # Sample random minibatch
        rand!(rng, 𝒟, buffer, i = i)
        
        # Compute the gradient of the loss
        add_fisher_information_diagonal!(F, () -> -loss(𝒟), θ, i + N)
    end
    F, N + Nbatches
end

