## Generic deterministic network for values or policies
@with_kw mutable struct DeterministicNetwork <: Policy
    N
    output_dim = size(last(N.layers).b)
    device = device(N)
end

DeterministicNetwork(N, output_dim = size(last(N.layers).b); kwargs...) = DeterministicNetwork(N = N, output_dim = output_dim; kwargs...)

Flux.trainable(π::DeterministicNetwork) = Flux.trainable(π.N)

action_space(π::DeterministicNetwork) = ContinuousSpace(π.output_dim)
POMDPs.action(π::DeterministicNetwork, s) = mdcall(π.N, s, π.device)

POMDPs.value(π::DeterministicNetwork, s::AbstractArray) = mdcall(π.N, s, π.device)
POMDPs.value(π::DeterministicNetwork, s::AbstractArray, a::AbstractArray) = mdcall(π.N, vcat(s,a), π.device)

## DDPGPolicy
@with_kw mutable struct DDPGPolicy <: Policy
    A # actor 
    C # critic
    action_dim = size(last(A.layers).b)
    A⁻ = deepcopy(A)# target actor 
    C⁻ = deepcopy(C)# target critic
    device = device(A)
end

DDPGPolicy(A, C; kwargs...) = DDPGPolicy(A=A, C=C; kwargs...)

Flux.trainable(π::DDPGPolicy) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

POMDPs.value(π::DDPGPolicy, s, a) = mdcall(π.C, vcat(s,a), π.device)
target_value(π::DDPGPolicy, s, a) = mdcall(π.C⁻, vcat(s,a), π.device)

POMDPs.action(π::DDPGPolicy, s::AbstractArray) = mdcall(π.A, s, π.device)
target_action(π::DDPGPolicy, s::AbstractArray) = mdcall(π.A⁻, s, π.device)

action_space(π::DDPGPolicy) = ContinuousSpace(π.action_dim)

function update_target!(π::DDPGPolicy, τ = 1f0)
    polyak_average!(π.A⁻, π.A, τ)
    polyak_average!(π.C⁻, π.C, τ)
end

## Deep Q-network Policy
@with_kw mutable struct DQNPolicy <: Policy
    Q
    actions::Vector
    device = device(Q)
    Q⁻ = deepcopy(Q)
end

DQNPolicy(Q, actions::Vector; kwargs...) = DQNPolicy(Q = Q, actions = actions; kwargs...)

Flux.trainable(π::DQNPolicy) = Flux.trainable(π.Q)

POMDPs.action(π::DQNPolicy, s::S) where S <: AbstractArray = π.actions[argmax(value(π, s))]

POMDPs.value(π::DQNPolicy, s::S) where S <: AbstractArray = mdcall(π.Q, s, π.device)
POMDPs.value(π::DQNPolicy, s::AbstractArray, a::AbstractArray) = sum(value(π, s) .* a, dims = 1)

target_value(π::DQNPolicy, s::S) where S <: AbstractArray = mdcall(π.Q⁻, s, π.device)
target_value(π::DQNPolicy, s::AbstractArray, a::AbstractArray) = sum(target_vale(Q, s) .* a, dims = 1)

action_space(π::DQNPolicy) = DiscreteSpace(length(π.actions))

update_target!(π::DQNPolicy, τ = 1f0) = polyak_average!(π.Q⁻, π.Q, τ)

## Actor Critic Architecture
@with_kw mutable struct ActorCritic <: Policy
    A # actor 
    C # critic
end

ActorCritic(A, C::Chain) = ActorCritic(A, DeterministicNetwork(C))

Flux.trainable(π::ActorCritic) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

POMDPs.value(π::ActorCritic, s) = value(π.C, s)
POMDPs.value(π::ActorCritic, s, a) = value(π.C, s, a)

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

CategoricalPolicy(A, actions::Vector; kwargs...) = CategoricalPolicy(A = A, actions = actions; kwargs...)

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

GaussianPolicy(μ, logΣ; kwargs...) = GaussianPolicy(μ = μ, logΣ = logΣ; kwargs...)

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


## Exploration policy with Gaussian noise
@with_kw mutable struct GaussianNoiseExplorationPolicy <: ExplorationPolicy
    σ::Function = (i) -> 0.01f0
    rng::AbstractRNG = Random.GLOBAL_RNG
end

GaussianNoiseExplorationPolicy(σ::Real, rng::AbstractRNG = Random.GLOBAL_RNG) = GaussianNoiseExplorationPolicy((i) -> σ, rng)
GaussianNoiseExplorationPolicy(σ::Function; kwargs...) = GaussianNoiseExplorationPolicy(σ = (i) -> σ; kwargs...)

function POMDPs.action(π::GaussianNoiseExplorationPolicy, on_policy::Union{Policy, Chain}, k, s::AbstractArray)
    a = action(on_policy, s)
    ϵ = randn(π.rng, length(a))*π.σ(k)
    return a + ϵ
end


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

