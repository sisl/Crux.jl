@with_kw struct BufferParams
    init::Int64 = 0
    size::Int64
end

mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    elements::Int64
    next_ind::Int64
end

function ExperienceBuffer(sdim::Int, adim::Int, size::Int, Nelements::Int = 0; device = cpu)
    Atype = device == gpu ? CuArray{Float32,2} : Array{Float32,2}
    ExperienceBuffer( Dict(:s => Atype(undef, sdim, size), 
                           :a => Atype(undef, adim, size), 
                           :sp => Atype(undef, sdim, size), 
                           :r => Atype(undef, 1, size), 
                           :done => Atype(undef, 1, size), 
                           :logprob => Atype(undef, 1, size)),
                           Nelements, 1)
end

ExperienceBuffer(mdp, size::Int, Nelements::Int = 0; device = cpu) = ExperienceBuffer(sdim(mdp), adim(mdp), size, Nelements, device = device)

function ExperienceBuffer(mdp, policy, init::Int, N::Int; desired_return = nothing, max_tries = 100*N, max_steps = 100, rng::AbstractRNG = Random.GLOBAL_RNG)
    b = ExperienceBuffer(mdp, N)
    i = 0
    while length(b) < init && (i += 1) < max_tries
        h = simulate(HistoryRecorder(max_steps = max_steps, rng = rng), mdp, policy)
        if isnothing(desired_return) || undiscounted_reward(h) ≈ desired_return
            [push!(b, s, a, r, sp, isterminal(mdp, sp), mdp) for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")]
        end
    end
    b
end

Base.getindex(b::ExperienceBuffer, key) = view(b.data[key], :, 1:b.elements)

Base.length(b::ExperienceBuffer) = b.elements

empty_like(b::ExperienceBuffer) = ExperienceBuffer(size(b.data[:s],1), size(b.data[:a], 1), size(b.data[:s], 2))

function Base.push!(b::ExperienceBuffer, s, a, r, sp, done, mdp)
    b.data[:s][:, b.next_ind] .= convert_s(AbstractVector, s, mdp) 
    b.data[:a][:, b.next_ind] .= Flux.onehot(a, actions(mdp))
    b.data[:sp][:, b.next_ind] .= convert_s(AbstractVector, sp, mdp)
    b.data[:r][1, b.next_ind] = r
    b.data[:done][1, b.next_ind] = done
    b.elements = min(length(b.data[:r]), b.elements + 1)
    b.next_ind = mod1(b.next_ind + 1,  length(b.data[:r]))
end

function Base.push!(b::ExperienceBuffer, mdp, s, a; rng::AbstractRNG = Random.GLOBAL_RNG)
    sp, r = gen(mdp, s, a, rng)
    done = isterminal(mdp, sp)
    push!(b, s, a, r, sp, done, mdp)
    done ? rand(rng, initialstate(mdp)) : sp 
end

Base.push!(b::ExperienceBuffer, mdp, s, policy::Policy; rng::AbstractRNG = Random.GLOBAL_RNG) = push!(b, mdp, s, action(policy, s), rng = rng)

Base.push!(b::ExperienceBuffer, mdp, s, policy::Policy, exploration_policy::Policy, i::Int; rng::AbstractRNG = Random.GLOBAL_RNG) = push!(b, mdp, s, action(exploration_policy, policy, i, s), rng = rng)

function trim!(b::ExperienceBuffer)
    b.elements = b.next_ind -1
    b.next_ind = 1
    b.data[:s] = b.data[:s][:, 1:b.elements]
    b.data[:a] = b.data[:a][:, 1:b.elements]
    b.data[:r] = b.data[:r][:, 1:b.elements]
    b.data[:sp] = b.data[:sp][:, 1:b.elements]
    b.data[:done] = b.data[:done][:, 1:b.elements]
end

function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer)
    N = size(first(target.data)[2], 2)
    ids = rand(rng, 1:source.elements, N)
    for k in keys(target.data)
        copyto!(target[k], source[k][:, ids])
    end
end

