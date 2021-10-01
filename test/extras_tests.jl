using Crux
using Test
using Zygote
using Flux

##  MultitaskDecay Schedule
m = MultitaskDecaySchedule(10, [1,2,3])
l = Crux.LinearDecaySchedule(1.0, 0.1, 10)

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-20)
end

m = MultitaskDecaySchedule(10, [1,2,1])

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-10)
end

@test m(31) == 0.1
@test m(0) == 1

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

