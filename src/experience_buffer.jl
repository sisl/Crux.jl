const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

# Efficient inverse query for fenwick tree : adapted from https://codeforces.com/blog/entry/61364
function inverse_query(t::FenwickTree, v, N = length(t))
    tot, pos = 0, 0
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
function mdp_data(S::T1, A::T2, capacity::Int; ArrayType = Array, R = Float32, D = Bool, W = Float32, gae = false) where {T1 <: AbstractSpace, T2 <: AbstractSpace}
    data = Dict{Symbol, ArrayType}(
        :s => ArrayType(fill(zero(type(S)), dim(S)..., capacity)), 
        :a => ArrayType(fill(zero(type(A)), dim(A)..., capacity)), 
        :sp => ArrayType(fill(zero(type(S)), dim(S)..., capacity)), 
        :r => ArrayType(fill(zero(R), 1, capacity)), 
        :done => ArrayType(fill(zero(D), 1, capacity)),
        :weight => ArrayType(fill(one(W), 1, capacity)),
        )
    if gae
        data[:return] = ArrayType(fill(zero(R), 1, capacity))
        data[:advantage] = ArrayType(fill(zero(R), 1, capacity))
    end
    data
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

function ExperienceBuffer(S::T1, A::T2, capacity::Int; device = cpu, gae = false, 
                          prioritized = false, α = 0.6f0, β = (i) -> 0.5f0, max_priority = 1f0,
                          R = Float32, D = Bool, W = Float32) where {T1 <: AbstractSpace, T2 <: AbstractSpace}
    Atype = device == gpu ? CuArray : Array
    data = mdp_data(S, A, capacity, ArrayType = Atype, gae = gae,  R = R, D = D, W = W)
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
    data = Dict{Symbol, CuArray}(k => v |> gpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.minsort_priorities, b.priorities, b.α, b.β, b.max_priority)
end

function Flux.cpu(b::ExperienceBuffer)
    data = Dict{Symbol, Array}(k => v |> cpu for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.minsort_priorities, b.priorities, b.α, b.β, b.max_priority)
end

minibatch(b::ExperienceBuffer, indices) = Dict(k => bslice(b.data[k], indices) for k in keys(b))

Base.getindex(b::ExperienceBuffer, key::Symbol) = bslice(b.data[key], 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

Base.length(b::ExperienceBuffer) = b.elements

DataStructures.capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

prioritized(b::ExperienceBuffer) = !isnothing(b.priorities)

device(b::ExperienceBuffer{CuArray}) = gpu
device(b::ExperienceBuffer{Array}) = cpu

dim(b::ExperienceBuffer, s::Symbol) = size(b[s], 1)

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data; ids = nothing)
    ids = isnothing(ids) ? UnitRange(1, size(data[first(keys(data))], 2)) : ids
    N, C = length(ids), capacity(b)
    I = mod1.(b.next_ind:b.next_ind + N - 1, C)
    for k in keys(data)
        copyto!(bslice(b.data[k], I), collect(bslice(data[k], ids)))
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
    target.indices = [inverse_query(source.priorities, (j + rand(rng) - 1) * Δp, N-1) for j=1:B]
    target.indices = max.(target.indices, )
    pmin = first(source.minsort_priorities) / ptot
    max_w = (pmin*N)^(-source.β(i))
    source[:weight][1, target.indices] .= [(N * source.priorities[id] / ptot)^source.β(i) for id in target.indices] ./ max_w
    
    push!(target, source, ids = target.indices)
end

