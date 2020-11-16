## General GPU support for policies
Flux.params(π::Policy, device) = Flux.params(network(π, device)...)

function sync!(π::Policy, device)
    device == cpu && return 
    cpu_nets, gpu_nets = network(π, cpu),  network(π, gpu)
    for i=1:length(cpu_nets)
        copyto!(cpu_nets[i], gpu_nets[i])
    end
end

function Flux.Optimise.train!(π::Policy, loss::Function, opt, device)
    θ = Flux.params(π, device)
    loss, back = Flux.pullback(loss, θ)
    grad = back(1f0)
    Flux.update!(opt, θ, grad)
    sync!(π, device)
    loss, grad
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

function fill_gae!(b::ExperienceBuffer, start::Int, Nsteps::Int, V, λ::Float32, γ::Float32)
    A, c = 0f0, λ*γ
    for i in reverse(get_indices(b, start, Nsteps))
        Vsp = V(b[:sp][:,i])
        Vs = V(b[:s][:,i])
        @assert length(Vs) == 1
        A = c*A + b[:r][1,i] + (1.f0 - b[:done][1,i])*γ*Vsp[1] - Vs[1]
        b[:advantage][:, i] .= A
    end
end

function fill_returns!(b::ExperienceBuffer, start::Int, Nsteps::Int, γ::Float32)
    r = 0f0
    for i in reverse(get_indices(b, start, Nsteps))
        r = b[:r][i] + γ*r
        b[:return][:, i] .= r
    end
end

## Categorical Policy
mutable struct DQNPolicy <: Policy
    Q
    mdp
    Q_GPU
end

DQNPolicy(Q, mdp; device = cpu) = DQNPolicy(Q, mdp, todevice(Q, device))

network(π::DQNPolicy, device) = (device == gpu) ? [π.Q_GPU] : [π.Q]

POMDPs.action(π::DQNPolicy, s) = actions(π.mdp)[argmax(π.Q(convert_s(AbstractVector, s, π.mdp)))]

POMDPs.value(π::DQNPolicy, s::AbstractArray) = network(π, device(s))[1](s)

## Categorical Policy
mutable struct CategoricalPolicy <: Policy
    A
    mdp
    rng::AbstractRNG
    A_GPU
end

CategoricalPolicy(A, mdp; device = cpu, rng::AbstractRNG = Random.GLOBAL_RNG) = CategoricalPolicy(A, mdp, rng, todevice(A, device))

network(π::CategoricalPolicy, device) = (device == gpu) ? [π.A_GPU] : [π.A]

POMDPs.action(π::CategoricalPolicy, s) = actions(π.mdp)[rand(π.rng, Categorical(π.A(convert_s(AbstractVector, s, π.mdp))))]

logits(π::CategoricalPolicy, s::AbstractArray) = network(π, device(s))[1](s)
    
function Distributions.logpdf(π::CategoricalPolicy, s::AbstractArray, a::AbstractArray)
    log.(sum(logits(π, s) .* a, dims = 1) .+ eps(Float32))
end


## Gaussian Policy
@with_kw mutable struct GaussianPolicy <: Policy
    μ
    logΣ
    mdp
    rng::AbstractRNG = Random.GLOBAL_RNG
    μ_GPU = nothing
    logΣ_GPU = nothing
end

GaussianPolicy(μ, logΣ, mdp; rng::AbstractRNG = Random.GLOBAL_RNG) = GaussianPolicy(μ, logΣ, mdp, rng, todevice(μ, device), todevice(logΣ, device))

network(π::GaussianPolicy, device) = (device == gpu) ? [π.μ, π.logΣ] : [π.μ_GPU, π.logΣ_GPU]

function POMDPs.action(π::GaussianPolicy, s)
    svec = convert_s(AbstractVector, s, π.mdp)
    d = MvNormal(π.μ(svec), diagm(0=>exp.(π.logΣ).^2))
    rand(rng, d)
end

function Distributions.logpdf(π::GaussianPolicy, s::AbstractArray, a::AbstractArray)
    μ_net, logΣ_net = network(p, device(s))
    μ = μ_net(s)
    σ2 = exp.(logΣ_net).^2
    sum((a .- (μ ./ σ2)).^2 .- 0.5 * log.(6.2831853071794*σ2), dims=2)
end

