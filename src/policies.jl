abstract type NetworkPolicy <: Policy end
device(π::NetworkPolicy) = π.device

function polyak_average!(to, from, τ=1f0)
    to_data = Flux.params(to).order.data
    from_data, from_device = Flux.params(from).order.data, device(from)
    device_match = from_device == device(to)
    for i = 1:length(to_data)
        if device_match
            copyto!(to_data[i], τ.*from_data[i] .+ (1f0-τ).*to_data[i])
        else
            copyto!(to_data[i], τ.*from_data[i] .+ (1f0-τ).*from_device(to_data[i]))
        end            
    end
end

function Base.copyto!(to, from)
    for i = 1:length(Flux.params(to).order.data)
        copyto!(Flux.params(to).order.data[i], Flux.params(from).order.data[i])
    end
end

## Network for representing continous functions (value or policy)
@with_kw mutable struct ContinuousNetwork <: NetworkPolicy
    network
    output_dim = size(last(network.layers).b)
    device = device(network)
end

ContinuousNetwork(network, output_dim = size(last(network.layers).b)) = ContinuousNetwork(network = network, output_dim = output_dim)

Flux.trainable(π::ContinuousNetwork) = Flux.trainable(π.network)

POMDPs.action(π::ContinuousNetwork, s) = value(π, s)

POMDPs.value(π::ContinuousNetwork, s::AbstractArray) = mdcall(π.network, s, π.device)
POMDPs.value(π::ContinuousNetwork, s::AbstractArray, a::AbstractArray) = mdcall(π.network, vcat(s,a), π.device)

action_space(π::ContinuousNetwork) = ContinuousSpace(π.output_dim)


## Network for representing a discrete set of outputs (value or policy)
@with_kw mutable struct DiscreteNetwork <: NetworkPolicy
    network
    outputs::Vector
    device = device(network)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

DiscreteNetwork(network, outputs::Vector; kwargs...) = DiscreteNetwork(network = network, outputs = outputs; kwargs...)
Flux.trainable(π::DiscreteNetwork) = Flux.trainable(π.network)

POMDPs.action(π::DiscreteNetwork, s::S) where S <: AbstractArray = π.outputs[argmax(value(π, s))] # Deterministic
POMDPs.action(π::DiscreteNetwork, on_policy::Policy, k, s::AbstractArray) = π.outputs[rand(π.rng, Categorical(value(π, s)[:]))] # Stochastic

POMDPs.value(π::DiscreteNetwork, s::S) where S <: AbstractArray = mdcall(π.network, s, π.device)
POMDPs.value(π::DiscreteNetwork, s::AbstractArray, a::AbstractArray) = sum(value(π, s) .* a, dims = 1)


action_space(π::DiscreteNetwork) = DiscreteSpace(length(π.outputs))

function logpdf(π::DiscreteNetwork, s::AbstractArray, a::AbstractArray)
    log.(sum(value(π, s) .* a, dims = 1) .+ eps(Float32))
end

function entropy(π::DiscreteNetwork, s::AbstractArray)
    aprob = value(π, s)
    sum(aprob .* log.(aprob .+ eps(Float32)), dims=1)
end



## Actor Critic Architecture
@with_kw mutable struct ActorCritic{TA, TC} <: NetworkPolicy
    A::TA # actor 
    C::TC # critic
end

device(π::ActorCritic) = device(π.A)

Flux.trainable(π::ActorCritic) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

POMDPs.value(π::ActorCritic, s) = value(π.C, s)
POMDPs.value(π::ActorCritic, s, a) = value(π.C, s, a)

POMDPs.action(π::ActorCritic, s::AbstractArray) = action(π.A, s)
POMDPs.action(π::ActorCritic, on_policy::Policy, k, s::AbstractArray) = action(π.A, on_policy, k, s)
    
logpdf(π::ActorCritic, s::AbstractArray, a::AbstractArray) = logpdf(π.A, s, a)

action_space(π::ActorCritic) = action_space(π.A)

entropy(π::ActorCritic, s::AbstractArray) = entropy(π.A, s)


## Gaussian Policy
@with_kw mutable struct GaussianPolicy <: NetworkPolicy
    μ::ContinuousNetwork
    logΣ::AbstractArray
    device = device(μ)
    rng::AbstractRNG = Random.GLOBAL_RNG
end

GaussianPolicy(μ, logΣ; kwargs...) = GaussianPolicy(μ = μ, logΣ = logΣ; kwargs...)

Flux.trainable(π::GaussianPolicy) = (Flux.trainable(π.μ)..., π.logΣ)

POMDPs.action(π::GaussianPolicy, s::AbstractArray) = action(π.μ, s)

function POMDPs.action(π::GaussianPolicy, on_policy::Policy, k, s::AbstractArray) 
    μ, logΣ = action(π, s), device(s)(π.logΣ)
    d = MvNormal(μ, exp.(logΣ))
    a = rand(π.rng, d)
end

function logpdf(π::GaussianPolicy, s::AbstractArray, a::AbstractArray)
    μ, logΣ = action(π, s), device(s)(π.logΣ)
    σ² = exp.(logΣ).^2
    sum(-((a .- μ).^2) ./ (2 .* σ²) .-  0.9189385332046727f0 .- logΣ, dims = 1) # 0.9189385332046727f0 = log.(sqrt(2π))
end

entropy(π::GaussianPolicy, s::AbstractArray) = 1.4189385332046727f0 .+ π.logΣ # 1.4189385332046727 = 0.5 + 0.5 * log(2π)

action_space(π::GaussianPolicy) = action_space(π.μ)


## Exploration policy with Gaussian noise
@with_kw mutable struct GaussianNoiseExplorationPolicy <: ExplorationPolicy
    σ::Function = (i) -> 0.01f0
    clip_min::Vector{Float32} = [-Inf32]
    clip_max::Vector{Float32} = [Inf32]
    rng::AbstractRNG = Random.GLOBAL_RNG
end

GaussianNoiseExplorationPolicy(σ::Real; kwargs...) = GaussianNoiseExplorationPolicy(σ = (i) -> σ; kwargs...)
GaussianNoiseExplorationPolicy(σ::Function; kwargs...) = GaussianNoiseExplorationPolicy(σ = (i) -> σ; kwargs...)

function POMDPs.action(π::GaussianNoiseExplorationPolicy, on_policy::Policy, k, s::AbstractArray)
    a = action(on_policy, s)
    ϵ = randn(π.rng, length(a))*π.σ(k)
    return clamp.(a + ϵ, π.clip_min, π.clip_max)
end


## use exploration policy for first N timesteps, then revert to base policy
@with_kw mutable struct FirstExplorePolicy <: ExplorationPolicy
    N::Int64 # Number of steps to explore for
    initial_policy::Policy # the policy to use for the first N steps
    after_policy::Union{Nothing, ExplorationPolicy} = nothing # the policy to use after the first N steps. Nothing means you will use on-policy
end

FirstExplorePolicy(N::Int64, initial_policy::Policy) = FirstExplorePolicy(N, initial_policy, after_policy)

function POMDPs.action(π::FirstExplorePolicy, on_policy::Policy, k, s::AbstractArray)
    if k < π.N
        return action(π.initial_policy, s)
    elseif isnothing(π.after_policy)
        return action(on_policy, s)
    else
        return action(π.after_policy, on_policy, k, s)
    end
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

