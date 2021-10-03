using Crux
using Test
using Zygote
using Flux

## gradient penalty
m = Dense(2,1, init=ones, bias=false)
x = ones(Float32, 2, 100)
@test gradient_penalty(m, x) ≈ (sqrt(2) - 1)^2


# Other gradient penalty test


idim = 5
batch_size = 8
m = Chain(Dense(idim, 2*idim, tanh), Dense(2*idim,1)) |> gpu
x = rand(Float32, idim, batch_size) |> gpu
y = rand(Float32, 1, batch_size) |> gpu

function total_loss()
    Flux.mse(m(x), y) + gradient_penalty(m, x)
end

l, b = Flux.pullback(total_loss, Flux.params(m))
grad = b(1f0)
grad.grads

