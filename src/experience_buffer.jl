const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

@with_kw mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    capacity::Int64
    next_ind::Int64 = 1
    
    indices::Vector{Int} = []
    priorities::Union{Nothing, MinHeap} = nothing
    priority_sums::Union{Nothing, FenwickTree} = nothing
    α::Float64 = 0.6
    β::Function = (i) -> 0.5
    max_priority::Float64 = 1.0
end

function mdp_data(sdim::Int, adim::Int, size::Int; Atype = Array{Float32,2}, gae = false)
    data = Dict(
        :s => Atype(undef, sdim, size), 
        :a => Atype(undef, adim, size), 
        :sp => Atype(undef, sdim, size), 
        :r => Atype(undef, 1, size), 
        :done => Atype(undef, 1, size),
        )
    if gae
        b.data[:return] = Atype(undef, 1, size)
        b.data[:advantage] = Atype(undef, 1, size)
    end
end
    
function ExperienceBuffer(sdim::Int, adim::Int, capacity::Int; device = cpu, gae = false, prioritized = false)
    Atype = device == gpu ? CuArray{Float32,2} : Array{Float32,2}
    data = mdp_data(sdim, adim, capacity, Atype = Atype, gae = gae)
    b = ExperienceBuffer(data = data, capacity = capacity)
    if prioritized
        b.priorities = MinHeap(zeros(Float32, capacity))
        b.priority_sums = FenwickTree(zeros(Float32, capacity))
    end
    b
end

function ExperienceBuffer(b::ExperienceBuffer; device = device(b))
    data = Dict(k => todevice(v, device) for (k,v) in b.data)
    ExperienceBuffer(data, b.capacity, b.next_ind, b.indices, b.priorities, b.priority_sums)
end

Base.getindex(b::ExperienceBuffer, key::Symbol) = view(b.data[key], :, 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

Base.length(b::ExperienceBuffer) = b.elements

capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

prioritized(b::ExperienceBuffer) = !isnothing(b.priorities)

function circular_indices(start, Nsteps, l)
    stop = mod1(start+Nsteps-1, l)
    Nsteps > l && (start = stop+1) # Handle overlap
    (stop >= start) ? collect(start:stop) : [start:l..., 1:stop...]
end

device(b::ExperienceBuffer{CuArray{Float32, 2}}) = gpu
device(b::ExperienceBuffer{Array{Float32, 2}}) = cpu

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data)
    N, C = size(data[first(keys(data))], 2), capacity(b)
    I = circular_indices(b.next_ind, N, C)
    for k in keys(data)
        b.data[k][:, I] .= data[k]
    end
    prioritized(b) && update_priorties!(b, I, MAX_PRIORITY*ones(N))
        
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
end

function update_priorities!(b, I::AbstractArray, v::AbstractArray)
    for i=1:I
        update!(b.priorities, i, v[i]^b.α)
        update!(b.priority_sums, i, v[i]^b.α)
        b.max_priority = max(v[i], b.max_priority)
    end
end

function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer; i = 1)
    prioritized(source) ? prioritized_sample!(target, source, rng, i=i) : uniform_sample!(target, source, rng)
end

function uniform_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG;)
    @assert capacity(target) <= length(source)
    N = capacity(target)
    ids = rand(rng, 1:source.elements, N)
    for k in keys(target.data)
        copyto!(target.data[k], source[k][:, ids])
    end
end

# With guidance from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
function prioritized_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG; i = 1)
    N, B = length(source), length(target)
    ptot = prioritized_sum[N]
    Δp = ptot / B
    ids = [inverse_query(source.priority_sums, (j + rand(rng)) * Δp) for j=1:B]
    pmin = first(source.priorities) / ptot
    max_w = (p_min*N)^source.β(i)
    ws = [(N * source.priorities[id] / ptot)^source.β(i) for id in ids] ./ max_w
    
    # Add the indices to the target and the weights to the target
    target.indices = ids
    for k in keys(target.data)
        copyto!(target.data[k], source[k][:, ids])
    end
    !haskey(target.data, :weight) && (target.data[:weight] = deepcopy(target.data[:r]))
    copyto!(target.data[:weight], ws)
end

function fillto!(b::ExperienceBuffer, s::Sampler, N::Int; i = 1)
    Nfill = max(0, N - length(b))
    Nfill > 0 && push!(b, steps!(s, i = i, Nsteps = Nfill))
    Nfill
end
