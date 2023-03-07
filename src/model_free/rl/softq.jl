# Soft-Q is technically on-policy, as the learned Q maps to the exploration policy. As such we have two options:
# 1) use a DiscreteNetwork for the main policiy, and define our own wrapped exploration function for the exploration policy
# 2) define a SoftDiscreteNetwork and use this for POMDPs.action() and exploration() [Preferred]
# Will have to do something similar in the case of ContinuousNetwork for SoftActorCritic


## Network for representing a discrete set of outputs (value or policy)
# NOTE: Incoming actions (i.e. arguments) are all assumed to be one hot encoding. Outputs are discrete actions taken form outputs
mutable struct SoftDiscreteNetwork <: NetworkPolicy
    network
    outputs
    α
    device
    SoftDiscreteNetwork(network, outputs; α=1., dev=nothing) = new(network, cpu(outputs), α, device(network))
    SoftDiscreteNetwork(network, outputs, α, dev) = new(network, cpu(outputs), α, device(network))
end

#logit_conversion=(π, s) -> softmax(value(π, s)),

Flux.@functor SoftDiscreteNetwork

Flux.trainable(π::SoftDiscreteNetwork) = Flux.trainable(π.network)

layers(π::SoftDiscreteNetwork) = π.network.layers

function Flux.onehotbatch(π::SoftDiscreteNetwork, a)
    ignore_derivatives() do
        a_oh = Flux.onehotbatch(a[:] |> cpu, π.outputs) |> device(a)
        length(a) == 1 ? dropdims(a_oh, dims=2) : a_oh
    end
end

# Return Q(s,⋅)
POMDPs.value(π::SoftDiscreteNetwork, s) = mdcall(π.network, s, π.device)

# Return V(s)
soft_value(π::SoftDiscreteNetwork, s) = π.α .* logsumexp((value(π, s) ./ π.α), dims=1)

# Return Q(s,a)
POMDPs.value(π::SoftDiscreteNetwork, s, a_oh) = sum(value(π, s) .* a_oh, dims=1)

POMDPs.action(π::SoftDiscreteNetwork, s) = exploration(π, s)[1]

logits(π::SoftDiscreteNetwork, s) = value(π, s) ./ π.α
probs(π::SoftDiscreteNetwork, s) = softmax(logits(π, s))

function exploration(π::SoftDiscreteNetwork, s; kwargs...)
    ps = probs(π, s)
    ai = mapslices((v) -> rand(Categorical(v)), ps, dims=1)
    a = π.outputs[ai]
    a, categorical_logpdf(ps, Flux.onehotbatch(π, a))
end

function Distributions.logpdf(π::SoftDiscreteNetwork, s, a)
    # If a does not seem to be a one-hot encoding then we encode it
    ignore_derivatives() do
        size(a, 1) == 1 && (a = Flux.onehotbatch(π, a))
    end
    return categorical_logpdf(probs(π, s), a)
end

function Distributions.entropy(π::SoftDiscreteNetwork, s)
    ps = probs(π, s)
    -sum(ps .* log.(ps .+ eps(Float32)), dims=1)
end

action_space(π::SoftDiscreteNetwork) = DiscreteSpace(length(π.outputs), π.outputs)




########## 

# since explore is on (offpolicysolver), can just define our own function 
# a, log_probs = exploration(sampler.agent.π_explore, sampler.svec, π_on=sampler.agent.π, i=i)

# exploration: action(s) propto softmax(q(s)/alpha) 
# target = reward + (1-done)*gamma*v_target(sp)
# v(s) = alpha*logsumexp(q(s)/alpha)
# v_target(sp) = alpha*logsumexp(q_target(sp)/alpha) 
# update q(s, a) to target

function SoftQ_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* soft_value(π, 𝒟[:sp])
end

function SoftQ(;π::SoftDiscreteNetwork, 
          N::Int, 
          ΔN=4, 
          c_opt::NamedTuple=(;epochs=4), 
          log::NamedTuple=(;),
          c_loss=td_loss(),
          target_fn=SoftQ_target,
          prefix="",
          kwargs...)
OffPolicySolver(;agent=PolicyParams(π=π, π⁻=deepcopy(π)), 
                  log=LoggerParams(;dir="log/dqn", log...),
                  N=N,
                  ΔN=ΔN,
                  c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), c_opt...),
                  target_fn=target_fn,
                  kwargs...)
end 
    




