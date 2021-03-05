@with_kw mutable struct Sampler{P, V, T1 <: AbstractSpace, T2 <: AbstractSpace}
    mdp::P
    π::Policy
    S::T1 # State space
    A::T2 # Action space
    max_steps::Int = 100
    required_columns::Array{Symbol} = []
    γ::Float32 = discount(mdp)
    λ::Float32 = NaN32
    π_explore = nothing
    s::V = rand(initialstate(mdp)) # Current State
    svec::AbstractArray = initial_observation(mdp, s) # Current observation
    episode_length::Int64 = 0
    episode_checker::Function = (data, start, stop) -> true
end

Sampler(mdp, π::Policy, S, A = action_space(π); kwargs...) = Sampler(mdp = mdp, π = π, S = S, A = A; kwargs...)

Sampler(mdps::AbstractVector, π::Policy, S, A = action_space(π); kwargs...) = [Sampler(mdp = mdps[i], π = π, S = S, A = A; kwargs...) for i in 1:length(mdps)]

function reset_sampler!(sampler::Sampler)
    sampler.s = rand(initialstate(sampler.mdp))
    sampler.svec = initial_observation(sampler.mdp, sampler.s)
    sampler.episode_length = 0
end

function initial_observation(mdp, s)
    if mdp isa POMDP
        return convert_o(AbstractArray, rand(initialobs(mdp, s)), mdp)
    else
        return convert_s(AbstractArray, s, mdp)
    end
end

function terminate_episode!(sampler::Sampler, data, j)
    haskey(data, :episode_end) && (data[:episode_end][1,j] = true)
    ep = j - sampler.episode_length + 1 : j
    haskey(data, :advantage) && fill_gae!(data, ep, sampler.π, sampler.λ, sampler.γ)
    haskey(data, :return) && fill_returns!(data, ep, sampler.γ)
    reset_sampler!(sampler)
end
    
function step!(data, j::Int, sampler::Sampler; explore=false, i=0)
    a, logprob = explore ? exploration(sampler.π_explore, sampler.svec, π_on=sampler.π, i=i) : (action(sampler.π, sampler.svec), NaN)
    length(a) == 1 && (a = a[1]) # actions always come out as an array
    if sampler.mdp isa POMDP
        sp, o, r = gen(sampler.mdp, sampler.s, a)
        spvec = convert_o(AbstractArray, o, sampler.mdp)
    else
        sp, r = gen(sampler.mdp, sampler.s, a)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)
    end
    done = isterminal(sampler.mdp, sp)

    # Save the tuple
    bslice(data[:s], j) .= reshape(sampler.svec, dim(sampler.S)...)
    data[:a][:, j] .= a  
    bslice(data[:sp], j) .= reshape(spvec, dim(sampler.S)...)
    data[:r][1, j] = r
    data[:done][1, j] = done
    
    # Handle optional data storage
    haskey(data, :logprob) && (data[:logprob][:, j] .= logprob)  
    haskey(data, :t) && (data[:t][1, j] = sampler.episode_length + 1)
    
    # Cut the episode short if needed
    sampler.episode_length += 1
    if done || sampler.episode_length >= sampler.max_steps 
        terminate_episode!(sampler, data, j)
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end

function steps!(sampler::Sampler; Nsteps = 1, explore=false, i=0, reset=false, return_episodes=false)
    data = mdp_data(sampler.S, sampler.A, Nsteps, return_episodes ? [sampler.required_columns..., :episode_end] : sampler.required_columns)
    for j=1:Nsteps
        step!(data, j, sampler, explore=explore, i=i + (j-1))
    end
    reset && terminate_episode!(sampler, data, Nsteps)
    return_episodes ? (data, episodes(data)) : data
end

function steps!(samplers::Vector{T}; Nsteps = 1, explore = false, i = 0, reset = false, return_episodes = false) where {T<:Sampler}
    data = mdp_data(samplers[1].S, samplers[1].A, Nsteps*length(samplers), return_episodes ? [samplers[1].required_columns..., :episode_end] : samplers[1].required_columns)
    j = 1
    for s=1:Nsteps
        for sampler in samplers
            step!(data, j, sampler, explore = explore, i = i + (j-1))
            j += 1
        end
    end
    reset && terminate_episode!(sampler, data, Nsteps)
    return_episodes ? (data, episodes(data)) : data
end

function episodes!(sampler::Sampler; Neps=1, explore=false, i=0, return_episodes=false)
    reset_sampler!(sampler)
    data = mdp_data(sampler.S, sampler.A, Neps*sampler.max_steps, sampler.required_columns)
    episode_starts = zeros(Int, Neps)
    episode_ends = zeros(Int, Neps)
    j, k = 0, 1
    while k <= Neps
        episode_starts[k] = j+1
        while true
            j = j+1
            step!(data, j, sampler, explore=explore, i=i + (i-1))
            if sampler.episode_length == 0
                episode_ends[k] = j
                sampler.episode_checker(data, episode_starts[k], j) ? (k = k+1) : (j = episode_starts[k])
                break
            end
        end
    end
    trim!(data, j)
    return_episodes ? (data, zip(episode_starts, episode_ends)) : data
end

function fillto!(b::ExperienceBuffer, s::Union{Sampler, Vector{T}}, N::Int; i=1, explore=false) where {T <: Sampler}
    Nfill = max(0, N - length(b))
    Nfill > 0 && push!(b, steps!(s, i=i, Nsteps=Nfill, explore=explore))
    Nfill
end

## Undiscounted returns
undiscounted_return(data, start, stop) = sum(data[:r][1,start:stop])

function undiscounted_return(s::Sampler; Neps=100, kwargs...)
    data = episodes!(s, Neps=Neps; kwargs...)
    sum(data[:r]) / Neps
end


## Discounted returns
function discounted_return(data, start, stop, γ)
    r = 0f0
    for i in reverse(start:stop)
        r = data[:r][1, i] + γ*r
    end
    r
end

function discounted_return(s::Sampler; Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps=Neps, return_episodes=true; kwargs...)
    mean([discounted_return(data, e..., discount(s.mdp)) for e in episodes])
end

## Failures
failure(data, start, stop; threshold = 0.) = undiscounted_return(data, start, stop) < threshold

function failure(s::Sampler; threshold=0., Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true; kwargs...)
    mean([failure(data, e..., threshold = threshold) for e in episodes])
end
    
    
## Generalized Advantage Estimation
function fill_gae!(d::ExperienceBuffer, V, λ::Float32, γ::Float32)
    eps = episodes(d)
    for ep in eps
        fill_gae!(d, ep, V, λ, γ)
    end
end

function fill_gae!(d, episode_range, V, λ::Float32, γ::Float32)
    A, c = 0f0, λ*γ
    nd = ndims(d[:s])
    for i in reverse(episode_range)
        Vsp = value(V, bslice(d[:sp], i:i))
        Vs = value(V, bslice(d[:s], i:i))
        @assert length(Vs) == 1
        A = c*A + d[:r][1,i] + (1.f0 - d[:done][1,i])*γ*Vsp[1] - Vs[1]
        @assert !isnan(A)
        d[:advantage][:, i] .= A
    end
end

function fill_returns!(data, episode_range, γ::Float32)
    r = 0f0
    for i in reverse(episode_range)
        r = data[:r][1, i] + γ*r
        data[:return][:, i] .= r
    end
end

# Utils
function trim!(data::Dict{Symbol, Array}, N)
    for k in keys(data)
        data[k] = bslice(data[k], 1:N)
    end
    data
end

