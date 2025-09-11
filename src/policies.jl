"""
Struct for combining useful policy parameters together

```julia
    π::Pol
    space::T2 = action_space(π)
    π_explore = π
    π⁻ = nothing
    pa = nothing # nominal action distribution
```
"""
@with_kw mutable struct PolicyParams{Pol<:Policy,T2<:AbstractSpace}
    π::Pol
    space::T2 = action_space(π)
    π_explore = π
    π⁻ = nothing
    pa = nothing # nominal action distribution
end
PolicyParams(π::T) where {T<:Policy} = PolicyParams(π=π)

abstract type NetworkPolicy <: Policy end

# Fixing bug with gpu deepcopies
function Base.deepcopy(x::NetworkPolicy)
    invoke(deepcopy, Tuple{Any}, (x |> cpu)) |> device(x)
end

device(π::N) where {N<:NetworkPolicy} = π.device
actor(π::N) where {N<:NetworkPolicy} = π
critic(π::N) where {N<:NetworkPolicy} = π

new_ep_reset!(π::N) where {N<:Policy} = nothing

# Call policies as functions calls the value function
(pol::NetworkPolicy)(x...) = value(pol, x...)

# Trajecotry pdf
function trajectory_logpdf(π::N, D) where {N <: Policy}
	sum(logpdf(π, D[:s], D[:a]))
end

function trajectory_logpdf(π::N, D, ep) where {N <: Policy}
	sum(logpdf(π, D[:s][:, ep], D[:a][:, ep]))
end

layers(π::Chain) = π.layers

function polyak_average!(to, from, τ=1.0f0)
    to_data = Flux.params(to).order.data
    from_data, from_device = Flux.params(from).order.data, device(from)
    device_match = from_device == device(to)
    for i = 1:length(to_data)
        if device_match
            copyto!(to_data[i], τ .* from_data[i] .+ (1.0f0 - τ) .* to_data[i])
        else
            copyto!(to_data[i], τ .* from_data[i] .+ (1.0f0 - τ) .* from_device(to_data[i]))
        end
    end
end

function Base.copyto!(to, from)
    for i = 1:length(Flux.params(to).order.data)
        copyto!(Flux.params(to).order.data[i], Flux.params(from).order.data[i])
    end
end

## Network for representing continous functions (value or policy)
mutable struct ContinuousNetwork <: NetworkPolicy
    network
    output_dim
    device
    function ContinuousNetwork(network, output_dim=nothing, dev=nothing)
        if isnothing(output_dim)
            try
                # Flux v0.12+
                output_dim = size(last(network.layers).weight, 1)
            catch
                # Flux v0.11
                output_dim = size(last(network.layers).W, 1)
            end
        end
        new(network, output_dim, device(network))
    end
end

Flux.@functor ContinuousNetwork

Flux.trainable(π::ContinuousNetwork) = Flux.trainable(π.network)

layers(π::ContinuousNetwork) = π.network.layers

POMDPs.action(π::ContinuousNetwork, s) = value(π, s)

POMDPs.value(π::ContinuousNetwork, s) = mdcall(π.network, s, π.device)

POMDPs.value(π::ContinuousNetwork, s, a) = mdcall(π.network, vcat(s, a), π.device)

action_space(π::ContinuousNetwork) = ContinuousSpace(π.output_dim)



## Network for representing a discrete set of outputs (value or policy)
# NOTE: Incoming actions (i.e. arguments) are all assumed to be one hot encoding. Outputs are discrete actions taken form outputs
mutable struct DiscreteNetwork <: NetworkPolicy
    network
    outputs
    logit_conversion
    always_stochastic
    device
    DiscreteNetwork(network, outputs; logit_conversion=(π, s) -> softmax(value(π, s)), always_stochastic=false, dev=nothing) = new(network, cpu(outputs), logit_conversion, always_stochastic, device(network))
    DiscreteNetwork(network, outputs, logit_conversion, always_stochastic, dev) = new(network, cpu(outputs), logit_conversion, always_stochastic, device(network))
end

Flux.@functor DiscreteNetwork

Flux.trainable(π::DiscreteNetwork) = Flux.trainable(π.network)

layers(π::DiscreteNetwork) = π.network.layers

POMDPs.value(π::DiscreteNetwork, s) = mdcall(π.network, s, π.device)

POMDPs.value(π::DiscreteNetwork, s, a_oh) = sum(value(π, s) .* a_oh, dims=1)

POMDPs.action(π::DiscreteNetwork, s) = π.always_stochastic ? exploration(π, s)[1] : π.outputs[mapslices(argmax, value(π, s), dims=1)]

function Flux.onehotbatch(π::DiscreteNetwork, a)
    ignore_derivatives() do
        a_oh = Flux.onehotbatch(a[:] |> cpu, π.outputs) |> device(a)
        length(a) == 1 ? dropdims(a_oh, dims=2) : a_oh
    end
end

logits(π::DiscreteNetwork, s) = π.logit_conversion(π, s)

categorical_logpdf(probs, a_oh) = log.(sum(probs .* a_oh, dims=1))

function exploration(π::DiscreteNetwork, s; kwargs...)
    ps = logits(π, s)
    ai = mapslices((v) -> rand(Categorical(v)), ps, dims=1)
    a = π.outputs[ai]
    a, categorical_logpdf(ps, Flux.onehotbatch(π, a))
end

function Distributions.logpdf(π::DiscreteNetwork, s, a)
    # If a does not seem to be a one-hot encoding then we encode it
    ignore_derivatives() do
        size(a, 1) == 1 && (a = Flux.onehotbatch(π, a))
    end
    return categorical_logpdf(logits(π, s), a)
end

function Distributions.entropy(π::DiscreteNetwork, s)
    ps = logits(π, s)
    -sum(ps .* log.(ps .+ eps(Float32)), dims=1)
end

action_space(π::DiscreteNetwork) = DiscreteSpace(length(π.outputs), π.outputs)



## Double Network architecture
mutable struct DoubleNetwork{T1,T2} <: NetworkPolicy
    N1::T1
    N2::T2
end

Flux.@functor DoubleNetwork

Flux.trainable(π::DoubleNetwork) = (Flux.trainable(π.N1)..., Flux.trainable(π.N2)...)

layers(π::DoubleNetwork) = unique((layers(π.N1)..., layers(π.N2)...))

device(π::DoubleNetwork) = device(π.N1) == device(π.N2) ? device(π.N1) : error("Mismatched devices")

POMDPs.value(π::DoubleNetwork, s) = (value(π.N1, s), value(π.N2, s))

POMDPs.value(π::DoubleNetwork, s, a) = (value(π.N1, s, a), value(π.N2, s, a))

POMDPs.action(π::DoubleNetwork, s) = (action(π.N1, s), action(π.N2, s))

exploration(π::DoubleNetwork, s; kwargs...) = (exploration(π.N1, s; kwargs...), exploration(π.N2, s; kwargs...))

Distributions.logpdf(π::DoubleNetwork, s, a) = (logpdf(π.N1, s, a), logpdf(π.N2, s, a))

Distributions.entropy(π::DoubleNetwork, s) = (entropy(π.N1, s), entropy(π.N2, s))

action_space(π::DoubleNetwork) = action_space(π.N1)

## Mixture Model Network architecture (Mixed per action)
mutable struct MixtureNetwork <: NetworkPolicy
    networks::Array
    weights::ContinuousNetwork
	MixtureNetwork(networks::Array, weights::ContinuousNetwork) = new(networks, weights)
    MixtureNetwork(networks::Array, weights::AbstractArray) = new(networks, ContinuousNetwork(Chain(ConstantLayer(weights)), length(weights)))

end

Flux.@functor MixtureNetwork

Flux.trainable(π::MixtureNetwork) = (Iterators.flatten([Flux.trainable(n) for n in π.networks])..., Flux.trainable(π.weights)...)

layers(π::MixtureNetwork) = unique((Iterators.flatten([layers(n) for n in π.networks])..., layers(π.weights)...))

function device(π::MixtureNetwork)
    dev = device(π.networks[1])
    @assert all([device(n) == dev for n in π.networks])
    @assert dev == device(π.weights)
    dev
end

POMDPs.action(π::MixtureNetwork, s) = exploration(π, s)[1]

function exploration(π::MixtureNetwork, s; kwargs...)
    αs = π.weights(s)
    indices = ignore_derivatives() do 
		αi = mapslices(α -> rand(Categorical(α)), αs, dims=1)[:]
		indices = []
		for i=1:length(π.networks)
			push!(indices,  αi .== i)
		end
		indices
	end
    println("weights: ", αs, "indices: ", indices)
    a = hcat([exploration(d, s[:, i])[1] for (d, i) in zip(π.networks, indices)]...)
    
    return a, logpdf(π, s, a)
end

function Distributions.logpdf(π::MixtureNetwork, s, a)
    α = π.weights(s)
    
    x = log.(sum([α[i] .* exp.(logpdf(p, s, a)) for (i, p) in enumerate(π.networks)]))
    
    # x = vcat([logpdf(p, s, a) for p in π.networks]...)
    # weighted_logsumexp(x, α)
end

Distributions.entropy(π::MixtureNetwork, s) = @error "Entropy not defined"

function action_space(π::MixtureNetwork) 
    action_space(π.networks[1])
end


## Actor Critic Architecture
mutable struct ActorCritic{TA,TC} <: NetworkPolicy
    A::TA # actor 
    C::TC # critic
end

Flux.@functor ActorCritic

Flux.trainable(π::ActorCritic) = (Flux.trainable(π.A)..., Flux.trainable(π.C)...)

layers(π::ActorCritic) = unique((layers(π.A)..., layers(π.C)...))

device(π::ActorCritic) = device(π.A) == device(π.C) ? device(π.A) : error("Mismatched devices")

POMDPs.value(π::ActorCritic, s) = value(π.C, s)

POMDPs.value(π::ActorCritic, s, a) = value(π.C, s, a)

POMDPs.action(π::ActorCritic, s) = action(π.A, s)

exploration(π::ActorCritic, s; kwargs...) = exploration(π.A, s; kwargs...)

Distributions.logpdf(π::ActorCritic, s, a) = logpdf(π.A, s, a)

trajectory_logpdf(π::ActorCritic, D...) = trajectory_logpdf(π.A, D...)

Distributions.entropy(π::ActorCritic, s) = entropy(π.A, s)

action_space(π::ActorCritic) = action_space(π.A)

actor(π::AC) where {AC<:ActorCritic} = π.A
critic(π::AC) where {AC<:ActorCritic} = π.C


## Network for concatentaing a latent vector to states
mutable struct LatentConditionedNetwork <: NetworkPolicy
    policy
    z
    reset_fn
    LatentConditionedNetwork(policy, z, reset_fn=(π) -> nothing) = new(policy, z, reset_fn)
end

function new_ep_reset!(π::LatentConditionedNetwork)
    π.reset_fn(π)
end

device(π::LatentConditionedNetwork) = device(π.policy)

Flux.@functor LatentConditionedNetwork

Flux.trainable(π::LatentConditionedNetwork) = Flux.trainable(π.policy)

layers(π::LatentConditionedNetwork) = layers(π.policy)

POMDPs.action(π::LatentConditionedNetwork, s) = action(π.policy, vcat(π.z, s))
POMDPs.value(π::LatentConditionedNetwork, s, args...) = value(π.policy, vcat(π.z, s), args...)

exploration(π::LatentConditionedNetwork, s; kwargs...) = exploration(π.policy, vcat(π.z, s); kwargs...)
Distributions.logpdf(π::LatentConditionedNetwork, s, a) = logpdf(π.policy, vcat(π.z, s), a)

Distributions.entropy(π::LatentConditionedNetwork, s) = entropy(π.policy, vcat(π.z, s))


action_space(π::LatentConditionedNetwork) = action_space(π.policy)
actor(π::LatentConditionedNetwork) = actor(π.policy)
critic(π::LatentConditionedNetwork) = critic(π.policy)



## Gaussian Policy
mutable struct GaussianPolicy <: NetworkPolicy
    μ::ContinuousNetwork
    logΣ::ContinuousNetwork
    always_stochastic::Bool
    GaussianPolicy(μ::ContinuousNetwork, logΣ::ContinuousNetwork, always_stochastic=false) = new(μ, logΣ, always_stochastic)
    GaussianPolicy(μ::ContinuousNetwork, logΣ::AbstractArray, always_stochastic=false) = new(μ, ContinuousNetwork(Chain(ConstantLayer(logΣ)), length(logΣ)), always_stochastic)
end

Flux.@functor GaussianPolicy

Flux.trainable(π::GaussianPolicy) = (Flux.trainable(π.μ)..., Flux.trainable(π.logΣ)...)

layers(π::GaussianPolicy) = (layers(π.μ)..., layers(π.logΣ))

device(π::GaussianPolicy) = device(π.μ) == device(π.logΣ) ? device(π.μ) : error("Mismatched devices")

POMDPs.action(π::GaussianPolicy, s) = π.always_stochastic ? exploration(π, s)[1] : π.μ(s)

function gaussian_logpdf(μ, logΣ, a)
    σ² = exp.(logΣ) .^ 2
    sum(-((a .- μ) .^ 2) ./ (2 .* σ²) .- 0.9189385332046727f0 .- logΣ, dims=1) # 0.9189385332046727f0 = log(sqrt(2π))
end

function exploration(π::GaussianPolicy, s; kwargs...)
    μ, logΣ = π.μ(s), π.logΣ(s)
    σ = exp.(logΣ)
    ϵ = ignore_derivatives(() -> randn(Float32, size(μ)...) |> device(s))
    a = ϵ .* σ .+ μ
    a, gaussian_logpdf(μ, logΣ, a)
end

Distributions.logpdf(π::GaussianPolicy, s, a) = gaussian_logpdf(π.μ(s), π.logΣ(s), a)

Distributions.entropy(π::GaussianPolicy, s) = 1.4189385332046727f0 .+ sum(π.logΣ(s)) # 1.4189385332046727 = 0.5 + 0.5 * log(2π)

action_space(π::GaussianPolicy) = action_space(π.μ)



## Squashed Gaussian policy
mutable struct SquashedGaussianPolicy <: NetworkPolicy
    μ::ContinuousNetwork
    logΣ::ContinuousNetwork
    ascale::Float32
    always_stochastic::Bool
    SquashedGaussianPolicy(μ, logΣ, ascale=1.0f0, always_stochastic=false) = new(μ, logΣ, ascale, always_stochastic)
    SquashedGaussianPolicy(μ, logΣ::Array, ascale=1.0f0, always_stochastic=false) = new(μ, ContinuousNetwork(Chain(ConstantLayer(logΣ)), length(logΣ)), ascale, always_stochastic)
end

Flux.@functor SquashedGaussianPolicy

Flux.trainable(π::SquashedGaussianPolicy) = (Flux.trainable(π.μ)..., Flux.trainable(π.logΣ)...)

layers(π::SquashedGaussianPolicy) = unique((layers(π.μ)..., layers(π.logΣ)...))

device(π::SquashedGaussianPolicy) = device(π.μ) == device(π.logΣ) ? device(π.μ) : error("Mismatched devices")

POMDPs.action(π::SquashedGaussianPolicy, s) = π.always_stochastic ? exploration(π, s)[1] : π.ascale .* tanh.(π.μ(s))

function squashed_gaussian_σ(logΣ)
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    # logΣ = LOG_STD_MIN .+ 0.5f0 .* (LOG_STD_MAX - LOG_STD_MIN) * (logΣ .+ 1)
    logΣ = clamp.(logΣ, LOG_STD_MIN, LOG_STD_MAX)
    exp.(logΣ)
end

# a is  untanh'd 
function squashed_gaussian_logprob(μ, logΣ, a)
    σ² = squashed_gaussian_σ(logΣ) .^ 2
    sum(-((a .- μ) .^ 2) ./ (2 .* σ²) .- 0.9189385332046727f0 .- logΣ .- 2 * (log(2.0f0) .- a .- softplus.(-2 .* a)), dims=1)
end

function exploration(π::SquashedGaussianPolicy, s; kwargs...)
    μ, logΣ = π.μ(s), π.logΣ(s)
    σ = squashed_gaussian_σ(logΣ)
    ϵ = ignore_derivatives(() -> randn(Float32, size(μ)...) |> device(s))
    a_pretanh = ϵ .* σ .+ μ
    π.ascale .* tanh.(a_pretanh), squashed_gaussian_logprob(μ, logΣ, a_pretanh)
end

Distributions.logpdf(π::SquashedGaussianPolicy, s, a) = squashed_gaussian_logprob(π.μ(s), π.logΣ(s), atanh.(clamp.(a ./ π.ascale, -1.0f0 + 1.0f-5, 1.0f0 - 1.0f-5)))

Distributions.entropy(π::SquashedGaussianPolicy, s) = 1.4189385332046727f0 .+ sum(π.logΣ(s), dims=1) # 1.4189385332046727 = 0.5 + 0.5 * log(2π) #TODO: This doesn't account for the squash

action_space(π::SquashedGaussianPolicy) = action_space(π.μ)

## Distribution policy -> For state-independent policies
mutable struct DistributionPolicy{T} <: Policy
    distribution::T
end

new_ep_reset!(π::DistributionPolicy) = nothing

layers(π::DistributionPolicy) = ()

device(π::DistributionPolicy) = cpu
actor(π::DistributionPolicy) = π

function POMDPs.action(π::DistributionPolicy{T}, s) where {T<:ContinuousMultivariateDistribution}
    B = ndims(s) > 1 ? size(s)[end] : () # NOTE: This hack doesnt work for states with multiple dimensions (i.e. images)
    Float32.(rand(π.distribution, B...)) |> device(s)
end

function POMDPs.action(π::DistributionPolicy{T}, s) where {T<:ContinuousUnivariateDistribution}
    B = ndims(s) > 1 ? size(s)[end] : 1
    a = Float32.(rand(π.distribution, B))
    (length(a) > 1 ? reshape(a, 1, :) : a) |> device(s)
end

function POMDPs.action(π::DistributionPolicy{T}, s) where {T<:DiscreteUnivariateDistribution}
    B = ndims(s) > 1 ? size(s)[end] : 1
    a = rand(π.distribution, B)
    (length(a) > 1 ? reshape(a, 1, :) : a) |> device(s)
end

function Distributions.logpdf(π::DistributionPolicy{T}, s, a) where {T<:ContinuousMultivariateDistribution}
    ls = Float32.(logpdf(π.distribution, reshape(a, length(π.distribution), :)))
    (length(ls) > 1 ? reshape(ls, 1, :) : ls) |> device(s)
end

function Distributions.logpdf(π::DistributionPolicy{T}, s, a) where {T<:ContinuousUnivariateDistribution}
    ls = Float32.(logpdf.(π.distribution, a))
    (length(ls) > 1 ? reshape(ls, 1, :) : ls) |> device(s)
end

function Distributions.logpdf(π::DistributionPolicy{T}, s, a) where {T<:DiscreteUnivariateDistribution}
    # If a seems to be a one-hot encoding then we onecold it
    size(a, 1) == length(support(π.distribution)) && (a = Flux.onecold(a, support(π.distribution)))

    ls = Float32.(logpdf.(π.distribution, a))
    (length(ls) > 1 ? reshape(ls, 1, :) : ls) |> device(s)
end

logits(π::DistributionPolicy{T}, s) where {T<:DiscreteUnivariateDistribution} = π.distribution.p

function exploration(π::DistributionPolicy{T}, s; kwargs...) where {T}
    a = action(π, s)
    a, logpdf(π, s, a)
end

function Distributions.entropy(π::DistributionPolicy, s)
    B = ndims(s) > 1 ? (size(s)[end]) : () # NOTE: this is a hack and doesn't work if the state space has more than one dim
    Float32(entropy(π.distribution)) * ones(Float32, 1, B...)
end

action_space(π::DistributionPolicy{T}) where {T<:ContinuousDistribution} = ContinuousSpace(length(π.distribution))
action_space(π::DistributionPolicy{T}) where {T<:DiscreteUnivariateDistribution} = DiscreteSpace(support(π.distribution))


## Mixed policy
mutable struct MixedPolicy <: Policy
    ϵ::Function
    policy
end

MixedPolicy(ϵ::Real, policy) = MixedPolicy((i) -> ϵ, policy)
ϵGreedyPolicy(ϵ, actions) = MixedPolicy(ϵ, DistributionPolicy(ObjectCategorical(actions)))

function exploration(π::MixedPolicy, s; π_on, i)
    ϵ = π.ϵ(i)
    x = (rand() < ϵ) ? exploration(π.policy, s)[1] : action(π_on, s) #exploration(π_on, s)[1]


    # Turn the action into an array if it is a value
    # !(x isa AbstractArray || x isa Tuple) && (x=fill(x, 1))

    # logp1 = Base.log(ϵ) .+ logpdf(π.policy, s, x)
    # logp2 = Base.log(1 - ϵ) .+ logpdf(π_on, s, x)

    p1 = ϵ .* exp.(logpdf(π.policy, s, x))
	if hasproperty(actor(π_on), :always_stochastic) && actor(π_on).always_stochastic
    	p2 = (1 - ϵ) .* exp.(logpdf(π_on, s, x))
	else
		p2 = (1 - ϵ)
	end

    # logsumexp(vcat(logp1, logp2), dims=1)
    x, Base.log.(p1 .+ p2)
end



## Exploration policy with Gaussian noise
@with_kw mutable struct GaussianNoiseExplorationPolicy <: Policy
    σ::Function = (i) -> 0.01f0
    a_min::Vector{Float32} = [-Inf32]
    a_max::Vector{Float32} = [Inf32]
    ϵ_min::Float32 = -Inf32
    ϵ_max::Float32 = Inf32
end

GaussianNoiseExplorationPolicy(σ::Real; kwargs...) = GaussianNoiseExplorationPolicy(σ=(i) -> σ; kwargs...)
GaussianNoiseExplorationPolicy(σ::Function; kwargs...) = GaussianNoiseExplorationPolicy(σ=σ; kwargs...)

function exploration(π::GaussianNoiseExplorationPolicy, s; π_on, i)
    a = action(π_on, s) |> cpu
    ϵ = randn(Float32, size(a)...) * π.σ(i)
    clamp.(a .+ clamp.(ϵ, π.ϵ_min, π.ϵ_max), π.a_min, π.a_max) |> device(s), NaN
end


## use exploration policy for first N timesteps, then revert to base policy
@with_kw mutable struct FirstExplorePolicy <: Policy
    N::Int64 # Number of steps to explore for
    initial_policy::Policy # the policy to use for the first N steps
    after_policy::Union{Nothing,Policy} = nothing # the policy to use after the first N steps. Nothing means you will use on-policy
end

FirstExplorePolicy(N::Int64, initial_policy::Policy) = FirstExplorePolicy(N, initial_policy, nothing)

function exploration(π::FirstExplorePolicy, s; π_on, i)
    if i < π.N
        return action(π.initial_policy, s), NaN
    elseif isnothing(π.after_policy)
        return action(π_on, s), NaN
    else
        return exploration(π.after_policy, s, π_on=π_on, i=i)
    end
end

# State Dependent Distribution Policy
struct StateDependentDistributionPolicy <: NetworkPolicy
    pa
    action_space
end

function POMDPs.action(policy::StateDependentDistributionPolicy, s)
    a_dist = policy.pa(s)
    a = rand(a_dist)
    return a
end

function action_space(policy::StateDependentDistributionPolicy)
    policy.action_space
end