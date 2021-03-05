mutable struct DiagonalFisherRegularizer
    F # Current average Fisher Diagonal
    N::Int # Number of gradients used
    λ::Float32 # Penalty Parameter
    θ⁻ # last set of params
end

DiagonalFisherRegularizer(θ, λ = 1) = DiagonalFisherRegularizer([zeros(Float32, size(p)) for p in θ], 0, λ, deepcopy(θ))

function (R::DiagonalFisherRegularizer)(π)
    θ = Flux.params(π)
    R.N == 0 && return 0f0
    nparams = length(θ)
    tot = 0f0
    for (p1, p2, i) in zip(θ, R.θ⁻, 1:nparams)
        tot += R.λ*mean(R.F[i].*(p1 .- p2).^2)
    end
    tot / nparams
end 

function add_fisher_information_diagonal!(R::DiagonalFisherRegularizer, neg_loss, θ)
    # Compute the gradient of the negative loss
    grads = gradient(neg_loss, θ)
    R.N += 1
    # The diagonal entries are the square of the gradients (keep a running average)
    for (p, i) in zip(θ, 1:length(θ))
        R.F[i] += (grads[p].^2 .- R.F[i]) ./ R.N
    end        
end

function update_fisher!(R::DiagonalFisherRegularizer, 𝒟, loss, θ, batch_size; i=0)
    shuffle!(𝒟)
    for i in partition(1:length(𝒟), batch_size)
        mb = minibatch(𝒟, i)
        add_fisher_information_diagonal!(R, () -> -loss(𝒟), θ)
    end
    R.θ⁻ = deepcopy(θ)
    nothing
end

