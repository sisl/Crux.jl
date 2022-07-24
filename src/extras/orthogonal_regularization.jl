@with_kw struct OrthogonalRegularizer
    β::Float32 = 1f-4 # regularization coefficient
end

function (R::OrthogonalRegularizer)(π)    
    reg = 0f0
    dev = ignore_derivatives(()->device(π))
    for l in ignore_derivatives(()->filter((l)->hasproperty(l, :weight), layers(π)))
        W = to2D(l.weight)
        prod = W' * W
        mat = ignore_derivatives(() -> ones(Float32, size(prod)...) .- Matrix{Float32}(I, size(prod)...) |> dev)
        reg += norm(prod .* mat)^2
    end
    R.β*reg
end

