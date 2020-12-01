## General GPU support for policies
Flux.params(π::Policy, device) = Flux.params(network(π, device)...)

device(π::Policy) = isnothing(network(π, gpu)[1]) ? cpu : gpu

function sync!(π::Policy, device)
    device == cpu && return 
    cpu_nets, gpu_nets = network(π, cpu),  network(π, gpu)
    for i=1:length(cpu_nets)
        copyto!(cpu_nets[i], gpu_nets[i])
    end
end

function Flux.Optimise.train!(π::Policy, loss::Function, opt, device)
    θ = Flux.params(π, device)
    l, back = Flux.pullback(loss, θ)
    grad = back(1f0)
    gnorm = norm(grad, p=Inf)
    Flux.update!(opt, θ, grad)
    sync!(π, device)
    l, gnorm
end

function Flux.Optimise.train!(π::Policy, loss::Function, 𝒟::ExperienceBuffer, B, opt, device; rng::AbstractRNG = Random.GLOBAL_RNG)
    losses, grads = [], []
    for i in partition(shuffle(rng, 1:length(𝒟)), B)
        mb = minibatch(𝒟, i)
        l, g = train!(π, ()->loss(mb), opt, device)
        push!(losses, l)
        push!(grads, g)
    end
    losses, grads
end


## Baseline
mutable struct Baseline <: Policy
    V
    L
    opt
    steps::Int
    λ::Float32
    V_GPU
end

Baseline(V; L = Flux.Losses.mse, opt = ADAM(1f-3), steps::Int = 40, λ::Float32 = 0.95f0, device = cpu) = Baseline(V, L, opt, steps, λ, todevice(V, device))

network(b::Baseline, device) = (device == gpu) ? [b.V_GPU] : [b.V]

POMDPs.value(b::Baseline, s) = network(b, device(s))[1](s)

function Flux.Optimise.train!(b::Baseline, 𝒟::ExperienceBuffer)
    θ = Flux.params(b, device(𝒟))
    data = Flux.Data.DataLoader((𝒟[:s], 𝒟[:return]), batchsize = length(𝒟))
    for i=1:b.steps
        train!((x,y) -> b.L(value(b, x), y), θ, data, b.opt)
    end
    sync!(b,  device(𝒟))
end
    



## Deep Q-network Policy
mutable struct DQNPolicy <: Policy
    Q
    actions
    Q_GPU
    Q⁻
end

DQNPolicy(Q, actions; device = cpu) = DQNPolicy(Q, actions, todevice(Q, device), deepcopy(Q) |> device)

network(π::DQNPolicy, device) = (device == gpu) ? [π.Q_GPU] : [π.Q]

POMDPs.action(π::DQNPolicy, s::S) where S <: AbstractArray = π.actions[argmax(value(π, s))]

POMDPs.value(π::DQNPolicy, s::S) where S <: AbstractArray = network(π, device(s))[1](s)


## Categorical Policy
mutable struct CategoricalPolicy <: Policy
    A
    actions
    rng::AbstractRNG
    A_GPU
end

CategoricalPolicy(A, actions; device = cpu, rng::AbstractRNG = Random.GLOBAL_RNG) = CategoricalPolicy(A, actions, rng, todevice(A, device))

network(π::CategoricalPolicy, device) = (device == gpu) ? [π.A_GPU] : [π.A]

POMDPs.action(π::CategoricalPolicy, s::AbstractArray) = π.actions[rand(π.rng, Categorical(π.A(s)))]

logits(π::CategoricalPolicy, s::AbstractArray) = network(π, device(s))[1](s)
    
function Distributions.logpdf(π::CategoricalPolicy, s::AbstractArray, a::AbstractArray)
    log.(sum(logits(π, s) .* a, dims = 1) .+ eps(Float32))
end


## Gaussian Policy
@with_kw mutable struct GaussianPolicy <: Policy
    μ
    logΣ
    rng::AbstractRNG = Random.GLOBAL_RNG
    μ_GPU = nothing
    logΣ_GPU = nothing
end

GaussianPolicy(μ, logΣ; rng::AbstractRNG = Random.GLOBAL_RNG) = GaussianPolicy(μ, logΣ, rng, todevice(μ, device), todevice(logΣ, device))

network(π::GaussianPolicy, device) = (device == gpu) ? [π.μ, π.logΣ] : [π.μ_GPU, π.logΣ_GPU]

function POMDPs.action(π::GaussianPolicy, s::AbstractArray)
    d = MvNormal(π.μ(s), diagm(0=>exp.(π.logΣ).^2))
    rand(rng, d)
end

function Distributions.logpdf(π::GaussianPolicy, s::AbstractArray, a::AbstractArray)
    μ_net, logΣ_net = network(p, device(s))
    μ = μ_net(s)
    σ2 = exp.(logΣ_net).^2
    sum((a .- (μ ./ σ2)).^2 .- 0.5 * log.(6.2831853071794*σ2), dims=2)
end

