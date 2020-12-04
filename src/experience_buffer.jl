const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

# Efficient inverse query for fenwick tree : adapted from https://codeforces.com/blog/entry/61364
function inverse_query(t::FenwickTree, v)
    tot, pos, N = 0, 0, length(t)
    for i=floor(Int, log2(N)):-1:0
        new_pos = pos + 1 << i
        if new_pos <= N && tot + t.bi_tree[new_pos] < v
            tot += t.bi_tree[new_pos]
            pos = new_pos
        end
    end
    pos + 1
end

Base.getindex(t::FenwickTree, i::Int) = prefixsum(t, i) - prefixsum(t, i-1)

DataStructures.update!(t::FenwickTree, i, v) = inc!(t, i, v - t[i])

# construction of common mdp data
function mdp_data(sdim::Int, adim::Int, size::Int; Atype = Array{Float32,2}, gae = false)
    data = Dict(
        :s => Atype(undef, sdim, size), 
        :a => Atype(undef, adim, size), 
        :sp => Atype(undef, sdim, size), 
        :r => Atype(undef, 1, size), 
        :done => Atype(undef, 1, size),
        :weight => Atype(undef, 1, size),
        )
    copyto!(data[:weight], ones(1, size))
    if gae
        data[:return] = Atype(undef, 1, size)
        data[:advantage] = Atype(undef, 1, size)
    end
    data
end

# Function to comput index ranges on a circular buffer
function circular_indices(start, Nsteps, l)
    stop = mod1(start+Nsteps-1, l)
    Nsteps > l && (start = stop+1) # Handle overlap
    (stop >= start) ? collect(start:stop) : [start:l..., 1:stop...]
end
    
## Experience Buffer stuff
@with_kw mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    elements::Int64 = 0
    next_ind::Int64 = 1
    
    indices::Vector{Int} = []
    minsort_priorities::Union{Nothing, MinHeap} = nothing
    priorities::Union{Nothing, FenwickTree} = nothing
    α::Float32 = 0.6
    β::Function = (i) -> 0.5f0
    max_priority::Float32 = 1.0
end

function ExperienceBuffer(data::Dict{Symbol, T}) where {T <: AbstractArray}
    elements = size(first(data)[2], 2)
    ExperienceBuffer(data = data, elements = elements)
end

function ExperienceBuffer(sdim::Int, adim::Int, capacity::Int; device = cpu, gae = false, prioritized = false, α = 0.6f0, β = (i) -> 0.5f0, max_priority = 1f0)
    Atype = device == gpu ? CuArray{Float32,2} : Array{Float32,2}
    data = mdp_data(sdim, adim, capacity, Atype = Atype, gae = gae)
    b = ExperienceBuffer(data = data)
    if prioritized
        b.minsort_priorities = MinHeap(fill(Inf32, capacity))
        b.priorities = FenwickTree(fill(0f0, capacity))
        b.α = α
        b.β = β
        b.max_priority = max_priority
    end
    b
end

function Flux.gpu(b::ExperienceBuffer)
    data = Dict(k => v |> gpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.minsort_priorities, b.priorities, b.α, b.β, b.max_priority)
end

function Flux.cpu(b::ExperienceBuffer)
    data = Dict(k => v |> cpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.minsort_priorities, b.priorities, b.α, b.β, b.max_priority)
end

minibatch(b::ExperienceBuffer, indices) = Dict(k => view(b.data[k], :, indices) for k in keys(b))

Base.getindex(b::ExperienceBuffer, key::Symbol) = view(b.data[key], :, 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

Base.length(b::ExperienceBuffer) = b.elements

DataStructures.capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

prioritized(b::ExperienceBuffer) = !isnothing(b.priorities)

device(b::ExperienceBuffer{CuArray{Float32, 2}}) = gpu
device(b::ExperienceBuffer{Array{Float32, 2}}) = cpu

sdim(b::ExperienceBuffer) = size(b[:s], 1)
adim(b::ExperienceBuffer) = size(b[:a], 1)

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data; ids = nothing)
    ids = isnothing(ids) ? UnitRange(1, size(data[first(keys(data))], 2)) : ids
    N, C = length(ids), capacity(b)
    I = circular_indices(b.next_ind, N, C)
    for k in keys(data)
        copyto!(view(b.data[k], :, I), data[k][:, ids])
    end
    prioritized(b) && update_priorities!(b, I, b.max_priority*ones(N))
        
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
end

function update_priorities!(b, I::AbstractArray, v::AbstractArray)
    for i = 1:length(I)
        val = v[i] + eps(Float32)
        update!(b.priorities, I[i], val^b.α)
        update!(b.minsort_priorities, I[i], val^b.α)
        b.max_priority = max(val, b.max_priority)
    end
end

function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer...; i = 1)
    lengths = [length.(source)...]
    batches = floor.(Int, capacity(target) .* lengths ./ sum(lengths))
    batches[1] += capacity(target) - sum(batches)
    
    for (b, B) in zip(source, batches)
        B == 0 && continue
        prioritized(b) ? prioritized_sample!(target, b, rng, i=i, B=B) : uniform_sample!(target, b, rng, B=B)
    end
end

function uniform_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG; B = capacity(target))
    ids = rand(rng, 1:length(source), B)
    push!(target, source, ids = ids)
end

# With guidance from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
function prioritized_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG; i = 1, B = capacity(target))
    N = length(source)
    ptot = prefixsum(source.priorities, N)
    Δp = ptot / B
    target.indices = [inverse_query(source.priorities, (j + rand(rng) - 1) * Δp) for j=1:B]
    pmin = first(source.minsort_priorities) / ptot
    max_w = (pmin*N)^(-source.β(i))
    source[:weight][1, target.indices] .= [(N * source.priorities[id] / ptot)^source.β(i) for id in target.indices] ./ max_w
    
    push!(target, source, ids = target.indices)
end

