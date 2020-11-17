@with_kw struct BufferParams
    init::Int64 = 0
    size::Int64
end

mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    elements::Int64
    next_ind::Int64
end

function ExperienceBuffer(sdim::Int, adim::Int, size::Int; device = cpu, gae = false, Nelements = 0)
    Atype = device == gpu ? CuArray{Float32,2} : Array{Float32,2}
    b = ExperienceBuffer( Dict(:s => Atype(undef, sdim, size), 
                           :a => Atype(undef, adim, size), 
                           :sp => Atype(undef, sdim, size), 
                           :r => Atype(undef, 1, size), 
                           :done => Atype(undef, 1, size)),
                           Nelements, 1)
    if gae
        b.data[:return] = Atype(undef, 1, size)
        b.data[:advantage] = Atype(undef, 1, size)
    end
    b
end

ExperienceBuffer(mdp, size::Int; device = cpu, gae = false, Nelements = 0) = ExperienceBuffer(sdim(mdp), adim(mdp), size, device = device, gae = gae, Nelements = Nelements)

empty_like(b::ExperienceBuffer; device = cpu) =  ExperienceBuffer(size(b[:s], 1), size(b[:a], 1), capacity(b), device = device, gae = haskey(b.data, :advantage))

Base.getindex(b::ExperienceBuffer, key) = view(b.data[key], :, 1:b.elements)

Base.length(b::ExperienceBuffer) = b.elements

capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

clear!(b::ExperienceBuffer) = b.elements, b.next_ind = 0,1

function get_indices(b, start, Nsteps)
    stop = mod1(start+Nsteps-1, length(b))
    Nsteps > length(b) && (start = stop+1) # Handle overlap
    (stop > start) ? collect(start:stop) : [start:length(b)..., 1:stop...]
end

device(b::ExperienceBuffer{CuArray{Float32, 2}}) = gpu
device(b::ExperienceBuffer{Array{Float32, 2}}) = cpu

isgpu(b::ExperienceBuffer{CuArray{Float32, 2}}) = true
isgpu(b::ExperienceBuffer{Array{Float32, 2}}) = false

function Base.push!(b::ExperienceBuffer, data)
    for (k,v) in data
        b.data[k][:, b.next_ind] .= v
    end
    b.elements = min(capacity(b), b.elements + 1)
    b.next_ind = mod1(b.next_ind + 1, capacity(b))
end

function Base.push!(b::ExperienceBuffer, s, a, r, sp, done, mdp; other = Dict())
    data = Dict(:s => convert_s(AbstractArray, s, mdp),
                :a => (a isa AbstractArray) ?  a : Flux.onehot(a, actions(mdp)),
                :sp => convert_s(AbstractArray, sp, mdp),
                :r => r, :done => done)
    push!(b, merge(data, other))    
end

function push_step!(b::ExperienceBuffer, mdp, s, a; rng::AbstractRNG = Random.GLOBAL_RNG)
    sp, r = gen(mdp, s, a, rng)
    done = isterminal(mdp, sp)
    push!(b, s, a, r, sp, done, mdp)
    done ? rand(rng, initialstate(mdp)) : sp 
end

push_step!(b::ExperienceBuffer, mdp, s, policy::Policy; rng::AbstractRNG = Random.GLOBAL_RNG) = push_step!(b, mdp, s, action(policy, s), rng = rng)

push_step!(b::ExperienceBuffer, mdp, s, policy::Policy, exploration_policy::Policy, i::Int; rng::AbstractRNG = Random.GLOBAL_RNG) = push_step!(b, mdp, s, action(exploration_policy, policy, i, s), rng = rng)

function Base.fill!(b_in::ExperienceBuffer, mdp, policy, N = capacity(b_in); rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100)
    b = isgpu(b_in) ? empty_like(b_in, device = cpu) : b_in
    clear!(b)
    γ = Float32(discount(mdp))
    while length(b) < N
        h = simulate(HistoryRecorder(max_steps = min(max_steps, N - length(b)), rng = rng), mdp, policy)
        start, Nsteps = b.next_ind, length(h)
        [push!(b, s, a, r, sp, isterminal(mdp, sp), mdp) for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")]
        if !isnothing(baseline)
            fill_gae!(b, start, Nsteps, baseline.V, baseline.λ, γ)
            fill_returns!(b, start, Nsteps, γ)
        end
    end
    isgpu(b_in) && copyto!(b_in, b) # Copy from cpu buffer to the given gpu buffer
end

function trim!(b::ExperienceBuffer)
    b.elements = b.next_ind -1
    b.next_ind = 1
    for k in b.data
        b.data[k] = b.data[k][:, 1:b.elements]
    end
end

function Base.copyto!(target::ExperienceBuffer, source::ExperienceBuffer)
    for k in keys(target.data)
        copyto!(target[k], source[k])
    end
end
    
function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer)
    @assert length(target) <= length(source)
    N = length(target)
    ids = rand(rng, 1:source.elements, N)
    for k in keys(target.data)
        copyto!(target[k], source[k][:, ids])
    end
end

function gen_buffer(mdps, pol, N; desired_return = nothing, max_tries = 100*N, max_steps = 100, nonzero_transitions_only = false)
    b = ExperienceBuffer(mdps[1], N)
    i = 1
    while length(b) < N && i < max_tries
        mdp = mdps[mod1(i, length(mdps))]
        h = simulate(HistoryRecorder(max_steps = max_steps), mdp, pol)
        if isnothing(desired_return) || undiscounted_reward(h) ≈ desired_return
            for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")
                if !nonzero_transitions_only || r != 0
                    push!(b, s, a, r, sp, isterminal(mdp, sp), mdp)
                end
            end
        end
        i += 1
    end
    b
end

