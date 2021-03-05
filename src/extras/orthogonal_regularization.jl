@with_kw struct OrthogonalRegularizer
    β::Float32 = 1f-4 # regularization coefficient
end

function (R::OrthogonalRegularizer)(π)
    reg = 0f0
    for l in layers(π)
        if l isa Dense
            prod = l.W' * l.W
            mat = ignore(() -> ones(Float32, size(prod)...) .- Matrix{Float32}(I, size(prod)...) |> device(π))
            reg += norm(prod .* mat)^2
        end
        #TODO: implement for conv
    end
    R.β*reg
end

