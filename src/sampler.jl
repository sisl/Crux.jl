@with_kw struct Sampler
    mdp
    π::Policy
    max_steps::Int
    exploration_policy::Union{ExplorationPolicy, Nothing} = nothing
    rng::AbstractRNG = Random.GLOBAL_RNG
    s = rand(rng, initialstate(mdp))
    svec::AbstractArray = convert_s(AbstractArray, s, mdp)
    episode_length::Int64 = 0
    episode_checker::Function = (data, start, stop) -> true
end

function trim!(data::Dict{Symbol, Array{Float32, 2}}, N)
    for k in keys(data)
        data[k] = data[k][:, 1:N]
    end
    data
end

explore(s::Sampler) = !isnothing(s.exploration_policy)

function step!(data, j::Int, sampler::Sampler; i = 0, baseline = nothing)
    a = explore(sampler) ? action(sampler.exploration_policy, sampler.π, i, sampler.svec) : action(sampler.π, sampler.svec)
    sp, r = gen(mdp, sampler.s, a, sampler.rng)
    done = isterminal(mdp, sp)

    # Save the tuple
    spvec = convert_s(AbstractArray, sp, mdp)
    data[:s][:, j] .= sampler.svec
    data[:a][:, j] .= (a isa AbstractArray) ?  a : Flux.onehot(a, actions(mdp))
    data[:sp][:, j] .= spvec
    data[:r][1, j] = r
    data[:done][1, j] = done
    
    # Cut the episode short if needed
    if done || sampler.episode_length >= sampler.max_steps 
        if !isnothing(baseline)
            start = j - sampler.episode_length
            fill_gae!(data, start, j, baseline.V, baseline.λ, γ)
            fill_returns!(data, start, j, γ)
        end
        sampler.s = rand(sampler.rng, initialstate(mdp))
        sampler.svec = convert_s(AbstractArray, sampler.s, mdp)
        sampler.episode_length = 0
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end

function steps!(sampler::Sampler; Nsteps = 1, i = 0, baseline = nothing)
    data = mdp_data(sdim(sampler.mdp), adim(sampler.mdp), Nsteps, gae = !isnothing(baseline))
    for j=1:Nsteps
        step!(data, j, sampler, i = i + (j-1), baseline = baseline)
    end
    data
end

function episodes!(sampler::Sampler; Neps = 1, i = 0, baseline = nothing, return_episodes = false)
    data = mdp_data(sdmin(sampler.mdp), admin(sampler.mdp), Neps*sampler.max_steps, gae = !isnothing(baseline))
    episode_starts = zeros(Int, Neps)
    episode_ends = zeros(Int, Neps)
    j, k = 1, 1
    while k <= Neps
        episode_starts[k] = j
        while true
            step!(data, j, sampler, explore = explore, i = i + (i-1), baseline = baseline)
            if data[:done][1, i]
                episode_checker(data, episode_starts[k], j) ? (k = k+1) : (j = episode_starts[k])
                break
            end
            j = j+1
        end
        episode_ends[k] = j
    end
    trim!(data, j)
    return_episodes ? (data, zip(episode_starts, episode_ends)) : data
end

## Undiscounted returns
undiscounted_return(data, start, stop) = sum(data[:r][1,start:stop])

function undiscounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data = episodes!(s, Neps = Neps)
    sum(data[:r])
end

## Discounted returns
function discounted_return(data, start, stop)
    r = 0f0
    for i in reverse(start:stop)
        r = data[:r][1, i] + γ*r
    end
    r
end

function discounted_return(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true)
    sum([discounted_return(data, e...) for e in episodes])
end

## Failures
failure(data, start, stop; threshold = 0.) = undiscounted_return(data, start, stop) < threshold

function failure(mdp, policy; Neps = 100, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    s = Sampler(mdp = mdp, π = policy, max_steps = max_steps, rng = rng)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true)
    sum([failure(data, e...) for e in episodes])
end
    
    
## Generalized Advantage Estimation
function fill_gae!(data, start::Int, stop::Int, V, λ::Float32, γ::Float32)
    A, c = 0f0, λ*γ
    for i in reverse(start:stop)
        Vsp = V(b[:sp][:,i])
        Vs = V(b[:s][:,i])
        @assert length(Vs) == 1
        A = c*A + b[:r][1,i] + (1.f0 - b[:done][1,i])*γ*Vsp[1] - Vs[1]
        b[:advantage][:, i] .= A
    end
end

function fill_returns!(data, start::Int, stop::Int, γ::Float32)
    r = 0f0
    for i in reverse(start:stop)
        r = b[:r][1, i] + γ*r
        b[:return][:, i] .= r
    end
end

