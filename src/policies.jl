## Deep Q-network Policy
@with_kw mutable struct DQNPolicy <: Policy
    Q
    actions::Vector
    device = device(Q)
    Q⁻ = deepcopy(Q)
end

Flux.trainable(π::DQNPolicy) = Flux.trainable(π.Q)

POMDPs.action(π::DQNPolicy, s::S) where S <: AbstractArray = π.actions[argmax(value(π, s))]

POMDPs.value(π::DQNPolicy, s::S) where S <: AbstractArray = mdcall(π.Q, s, π.device)

action_space(π::DQNPolicy) = DiscreteSpace(length(π.actions))

## Actor Critic Architecture
@with_kw mutable struct ActorCritic <: Policy
    A # actor 
    C # critic
end

Flux.trainable(π::ActorCritic) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

POMDPs.value(π::ActorCritic, s; kwargs...) = value(π.C, s; kwargs...)

POMDPs.action(π::ActorCritic, s::AbstractArray) = action(π.A, s)
    
logpdf(π::ActorCritic, s::AbstractArray, a::AbstractArray) = logpdf(π.A, s, a)

action_space(π::ActorCritic) = action_space(π.A)

entropy(π::ActorCritic, s::AbstractArray) = entropy(π.A, s)


## Categorical Policy
@with_kw mutable struct CategoricalPolicy <: Policy
    A
    actions
    device = device(A)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

Flux.trainable(π::CategoricalPolicy) = Flux.trainable(π.A)

POMDPs.action(π::CategoricalPolicy, s::AbstractArray) = π.actions[rand(π.rng, Categorical(logits(π, s)[:]))]

logits(π::CategoricalPolicy, s::AbstractArray) = mdcall(π.A, s, π.device)
    
function logpdf(π::CategoricalPolicy, s::AbstractArray, a::AbstractArray)
    log.(sum(logits(π, s) .* a, dims = 1) .+ eps(Float32))
end

function entropy(π::CategoricalPolicy, s::AbstractArray)
    aprob = logits(π, s)
    sum(aprob .* log.(aprob .+ eps(Float32)), dims=1)
end

action_space(π::CategoricalPolicy) = DiscreteSpace(length(π.actions))


## Gaussian Policy
@with_kw mutable struct GaussianPolicy <: Policy
    μ
    logΣ
    device = device(μ)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

Flux.trainable(π::GaussianPolicy) = (Flux.trainable(π.μ)..., π.logΣ)

function POMDPs.action(π::GaussianPolicy, s::AbstractArray)
    μ, logΣ = mdcall(π.μ, s, π.device), device(s)(π.logΣ)
    d = MvNormal(μ, exp.(logΣ))
    a = rand(π.rng, d)
end

function logpdf(π::GaussianPolicy, s::AbstractArray, a::AbstractArray)
    μ = mdcall(π.μ, s, π.device)
    logΣ = device(s)(π.logΣ)
    σ² = exp.(logΣ).^2
    sum(-((a .- μ).^2) ./ (2 .* σ²) .-  0.9189385332046727f0 .- logΣ, dims = 1) # 0.9189385332046727f0 = log.(sqrt(2π))
end

entropy(π::GaussianPolicy, s::AbstractArray) = 1.4189385332046727f0 .+ π.logΣ # 1.4189385332046727 = 0.5 + 0.5 * log(2π)

action_space(π::GaussianPolicy) = ContinuousSpace((length(π.logΣ),), typeof(cpu(π.logΣ)[1]))

## Linear Policy - Archived for now
# @with_kw mutable struct LinearBaseline <: Baseline
#     θ = nothing
#     featurize::Function = control_features
#     c::Float32 = eps(Float32) # regularization_ceoff
#     λ::Float32 = 0.95f0 # gae
#     device = cpu
# end
# 
# function control_features(s::AbstractArray; t::AbstractArray)
#     vcat(s, s.^2, t, t.^2, t.^3, ones(Float32, 1, size(s,2)))
# end
# 
# function Flux.Optimise.train!(b::LinearBaseline, 𝒟::ExperienceBuffer)
#     X = b.featurize(𝒟[:s], t = 𝒟[:t])
#     y = 𝒟[:return]
#     d, n = size(X)
#     A = X * X' ./ n + b.c*b.device(Matrix{typeof(X[1])}(I,d,d))
#     B = X * y' ./ n
#     b.θ = dropdims(pinv(A) * B, dims = 2)
# end
# 
# POMDPs.value(b::LinearBaseline, s; kwargs...) = b.θ' * b.featurize(s; kwargs...) 

