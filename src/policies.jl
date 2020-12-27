## General GPU support for policies

function Flux.Optimise.train!(π::Policy, loss::Function, opt, device; regularizer = (θ) -> 0)
    θ = Flux.params(π)
    l, back = Flux.pullback(() -> loss() + regularizer(θ), θ)
    grad = back(1f0)
    gnorm = norm(grad, p=Inf)
    Flux.update!(opt, θ, grad)
    l, gnorm
end

# Train with minibatches
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
@with_kw mutable struct Baseline <: Policy
    V
    L = Flux.Losses.mse
    opt = ADAM(1f-3)
    steps::Int = 40
    λ::Float32 = 0.95f0
    device = device(V)
end

Flux.params(b::Baseline) = Flux.params(b.V)

POMDPs.value(b::Baseline, s) = mdcall(b.V, s, b.device)

function Flux.Optimise.train!(b::Baseline, 𝒟::ExperienceBuffer)
    θ = Flux.params(b)
    data = Flux.Data.DataLoader((𝒟[:s], 𝒟[:return]), batchsize = length(𝒟))
    for i=1:b.steps
        train!((x,y) -> b.L(value(b, x), y), θ, data, b.opt)
    end
end
    



## Deep Q-network Policy
@with_kw mutable struct DQNPolicy <: Policy
    Q
    actions::Vector
    device = device(Q)
    Q⁻ = deepcopy(Q)
end

Flux.params(π::DQNPolicy) = Flux.params(π.Q)

POMDPs.action(π::DQNPolicy, s::S) where S <: AbstractArray = π.actions[argmax(value(π, s))]

POMDPs.value(π::DQNPolicy, s::S) where S <: AbstractArray = mdcall(π.Q, s, π.device)

action_space(π::DQNPolicy) = DiscreteSpace(length(π.actions))


## Categorical Policy
@with_kw mutable struct CategoricalPolicy <: Policy
    A
    actions
    device = device(A)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

Flux.params(π::CategoricalPolicy) = Flux.params(π.A)

POMDPs.action(π::CategoricalPolicy, s::AbstractArray) = π.actions[rand(π.rng, Categorical(logits(π, s)[:]))]

logits(π::CategoricalPolicy, s::AbstractArray) = mdcall(π.A, s, π.device)
    
function logpdf(π::CategoricalPolicy, s::AbstractArray, a::AbstractArray)
    log.(sum(logits(π, s) .* a, dims = 1) .+ eps(Float32))
end

action_space(π::CategoricalPolicy) = DiscreteSpace(length(π.actions))


## Gaussian Policy
@with_kw mutable struct GaussianPolicy <: Policy
    μ
    logΣ
    device = device(μ)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

Flux.params(π::GaussianPolicy) = Flux.params(π.μ, π.logΣ)

function POMDPs.action(π::GaussianPolicy, s::AbstractArray)
    μ, logΣ = mdcall(π.μ, s, π.device), device(s)(π.logΣ)
    d = MvNormal(μ, diagm(0=>exp.(logΣ).^2))
    a = rand(π.rng, d)
    @assert length(a) == 1
    a[1]
end

function logpdf(π::GaussianPolicy, s::AbstractArray, a::AbstractArray)
    μ = mdcall(π.μ, s, device)
    σ = exp.(π.logΣ)
    σ² = σ.^2
    broadcast(-, ((a .- μ).^2f0)./(2f0 .* σ²)) .-  0.4594692666f0 .- log.(σ)
end

action_space(π::GaussianPolicy) = ContinuousSpace((length(π.logΣ),), typeof(π.logΣ[1]))

