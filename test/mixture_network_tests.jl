using Crux
using Flux
using POMDPs
using Distributions

using Plots

function DeepGMM()
    base = Chain(Dense(2, 32, relu)) 
    mu1 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    logΣ1 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    # mu1 = ContinuousNetwork(ConstantLayer([1f0]), 1)
    # logΣ1 = ContinuousNetwork(ConstantLayer([log(1f0)]), 1)
    
    mu2 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    logΣ2 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    
    # mu2 = ContinuousNetwork(ConstantLayer([-1f0]), 1)
    # logΣ2 = ContinuousNetwork(ConstantLayer([log(1f0)]), 1)
    
    αs = ContinuousNetwork(Chain(base..., Dense(32, 2), softmax), 2)
    # αs = ContinuousNetwork(Chain(ConstantLayer([1f0, 1f0]), softmax), 2)
    MixtureNetwork([GaussianPolicy(mu1, logΣ1), GaussianPolicy(mu2, logΣ2)], αs)
end

function G()
    base = Chain(Dense(2, 32, relu)) 
    mu1 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    logΣ1 = ContinuousNetwork(Chain(base..., Dense(32, 1)))
    
    GaussianPolicy(mu1, logΣ1)
end

function plot_gmm(model, s, Npts=100)
    rd(v) = round(v, digits=2)
    αs = rd.(model.weights(s))
    m1 = rd(model.networks[1].μ(s)[1])
    σ1 = rd(exp(model.networks[1].logΣ(s)[1]))
    m2 = rd(model.networks[2].μ(s)[1])
    σ2 = rd(exp(model.networks[2].logΣ(s)[1]))
    
    lb = min(m1 - 3*σ1, m2 - 3*σ2)
    ub = max(m1 + 3*σ1, m2 + 3*σ2)
    
    yrange = range(lb, ub, length=Npts)
    py = [exp(logpdf(model, s, [y])[1]) for y in yrange]
    
    plot(yrange, py, title="$(αs[1]) ⋅ N($m1, $σ1) + $(αs[2]) ⋅ N($m2, $σ2) ")
end


target_dist = MixtureModel([Normal(1,0.2), Normal(-1, 1)], [0.5, 0.5])
plot(-3:0.1:3, x -> pdf(target_dist, x))


y = reshape(Float32.(rand(target_dist, 100_000)), 1, :)
x = 0.1f0 * ones(Float32, 2, 100_000)
d = Flux.Data.DataLoader((x, y), batchsize=1024)

n = DeepGMM()
loss(x,y) = -mean(logpdf(n, x, y))

loss(x, y)
plot_gmm(n, [0.1, 0.1])
plot!(-3:0.1:3, x -> pdf(target_dist, x))

logpdf(n, x, y)


opt = Adam()


evalcb() = @show(loss(x, y))
throttled_cb = Flux.throttle(evalcb, 1)
Flux.@epochs 100 Flux.train!(loss, Flux.params(n), d, opt, cb = throttled_cb)
loss(x, y)
plot_gmm(n, x[:, 1])
plot!(-3:0.1:3, x -> pdf(target_dist, x))
histogram!(y[:], normalize=true)

