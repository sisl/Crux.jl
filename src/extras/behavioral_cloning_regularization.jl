@with_kw mutable struct BCRegularizer
    buffers
    𝒟s
    λ::Float32
    loss
end

function BCRegularizer(buffers, batch_size::Int, device; λ::Float32=1f0, loss=(π, 𝒟) -> Flux.mse(value(π, 𝒟[:s]), 𝒟[:value]))
    𝒟s = [buffer_like(buff, capacity=batch_size, device=device) for buff in buffers]
    BCRegularizer(buffers, 𝒟s, λ, loss)
end

function (R::BCRegularizer)(π)
    # sample a random batch for each buffer
    for (𝒟, buffer) in zip(R.𝒟s, R.buffers)
        ignore(()->rand!(𝒟, buffer))
    end

    # Return the mean
    R.λ*mean([R.loss(π, 𝒟) for 𝒟 in R.𝒟s])
end

