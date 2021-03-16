@with_kw mutable struct BatchRegularizer
    buffers # Datasets to use in the regularization
    𝒟s = nothing# Batches of those datasets
    batch_size::Int # Batch size of the data sets
    λ::Float32 = 1f0# regularization coefficients
    loss # Loss function that takes args (π, 𝒟)
end

value_regularization(π, 𝒟) = Flux.mse(value(π, 𝒟[:s]), 𝒟[:value])
action_regularization(π, 𝒟) = Flux.mse(action(π, 𝒟[:s]), 𝒟[:a])
action_value_regularization(π, 𝒟) = Flux.mse(value(π, 𝒟[:s], 𝒟[:a]), 𝒟[:value])

function (R::BatchRegularizer)(π)
    # sample a random batch for each buffer
    ignore() do
        isnothing(R.𝒟s) && (R.𝒟s = [buffer_like(b, capacity=R.batch_size, device=device(π)) for b in buffers])
        for (𝒟, buffer) in zip(R.𝒟s, R.buffers)
            rand!(𝒟, buffer)
        end
    end

    # Return the mean
    R.λ*mean([R.loss(π, 𝒟) for 𝒟 in R.𝒟s])
end

