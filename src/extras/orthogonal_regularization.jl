@with_kw struct OrthogonalRegularizer
    β::Float32 = 1f-4 # regularization coefficient
end

function (R::OrthogonalRegularizer)(π)
    reg = 0f0
    for l in ignore(()->filter((l)->l isa Dense, layers(π)))
        prod = l.weight' * l.weight
        mat = ignore(() -> ones(Float32, size(prod)...) .- Matrix{Float32}(I, size(prod)...) |> device(π))
        reg += norm(prod .* mat)^2
    end
    #TODO: implement for conv
    R.β*reg
end

