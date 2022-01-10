struct DeepEnsemble
        models::Array
        DeepEnsemble(generator, N::Int) = new([generator() for _=1:N])
end

Flux.@functor DeepEnsemble 

Flux.trainable(m::DeepEnsemble) = (Flux.trainable(model) for model in m.models)

# Get the mean and variance estimate from each network individually
function individual_forward(m::DeepEnsemble, x)
        os = [m(x) for m in m.models]
        ndim = Int(size(os[1],1) / 2)
        μs = [o[1:ndim, :] for o in os]
        σ²s = [softplus.(o[ndim+1:end, :]) .+ 1f-3 for o in os]
        μs, σ²s
end

# Compute the prediction and uncertainty of the ensemble
function (m::DeepEnsemble)(x)
        μs, σ²s = individual_forward(m, x)
        t1 = [σ² .+ μ .^2 for (μ, σ²) in zip(μs, σ²s)]
        mean(μs), mean(t1) .- mean(μs).^2
end

# The equation for the gaussian logpdf
de_gaussian_logpdf(μ, σ², y) = -log.(σ²) ./ 2f0 .- (y .- μ).^2 ./ (2f0 .*  σ²)

# Logpdf of some some x and y pair according to the full ensemble
Distributions.logpdf(m::DeepEnsemble, x, y) = de_gaussian_logpdf(m(x)..., y)

# Gets the mean negative log pdf for each network
function training_loss(m::DeepEnsemble, x, y, weights = ones(Float32, size(y)...))
        μs, σ²s = individual_forward(m, x)
        mean([-mean(weights .* de_gaussian_logpdf(μ, σ², y)) for (μ, σ²) in zip(μs, σ²s)])
end


struct DeepClassificationEnsemble
        models::Array
        DeepClassificationEnsemble(generator, N::Int) = new([generator() for _=1:N])
end

Flux.@functor DeepClassificationEnsemble 

Flux.trainable(m::DeepClassificationEnsemble) = (Flux.trainable(model) for model in m.models)

# Get the mean and variance estimate from each network individually
function individual_forward(m::DeepClassificationEnsemble, x)
        ps = [softmax(m(x)) for m in m.models]
end

# Compute the prediction and uncertainty of the ensemble
function (m::DeepClassificationEnsemble)(x)
        ps = individual_forward(m, x)
        mean(ps)
end

# Logpdf of some some x and y pair according to the full ensemble
function Distributions.logpdf(m::DeepClassificationEnsemble, x, y)
        log.(sum(m(x) .* y, dims=1) .+ 1f-10)
end

# Gets the mean negative log pdf for each network
function training_loss(m::DeepClassificationEnsemble, x, y, weights = ones(Float32, size(y)...))
        ps = individual_forward(m, x)
        mean([Flux.Losses.crossentropy(p, y) for p in ps])
end
        
