const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

function mdp_data(sdim::Int, adim::Int, size::Int; Atype = Array{Float32,2}, gae = false)
    data = Dict(
        :s => Atype(undef, sdim, size), 
        :a => Atype(undef, adim, size), 
        :sp => Atype(undef, sdim, size), 
        :r => Atype(undef, 1, size), 
        :done => Atype(undef, 1, size),
        )
    if gae
        data[:return] = Atype(undef, 1, size)
        data[:advantage] = Atype(undef, 1, size)
    end
    data
end

function circular_indices(start, Nsteps, l)
    stop = mod1(start+Nsteps-1, l)
    Nsteps > l && (start = stop+1) # Handle overlap
    (stop >= start) ? collect(start:stop) : [start:l..., 1:stop...]
end
    

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

function ExperienceBuffer(b::ExperienceBuffer; device = device(b))
    data = Dict(k => v |> device for (k,v) in b.data)
    ExperienceBuffer(data, b.elements, b.next_ind, b.indices, b.minsort_priorities, b.priorities, b.α, b.β, b.max_priority)
end

Base.getindex(b::ExperienceBuffer, key::Symbol) = view(b.data[key], :, 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

Base.length(b::ExperienceBuffer) = b.elements

capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

prioritized(b::ExperienceBuffer) = !isnothing(b.priorities)

device(b::ExperienceBuffer{CuArray{Float32, 2}}) = gpu
device(b::ExperienceBuffer{Array{Float32, 2}}) = cpu

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data)
    N, C = size(data[first(keys(data))], 2), capacity(b)
    I = circular_indices(b.next_ind, N, C)
    for k in keys(data)
        b.data[k][:, I] .= data[k]
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

function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer; i = 1)
    prioritized(source) ? prioritized_sample!(target, source, rng, i=i) : uniform_sample!(target, source, rng)
end

function uniform_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG;)
    B = capacity(target)
    ids = rand(rng, 1:length(source), B)
    for k in keys(target.data)
        copyto!(target.data[k], source[k][:, ids])
    end
    target.elements = B
end

# With guidance from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
function prioritized_sample!(target::ExperienceBuffer, source::ExperienceBuffer, rng::AbstractRNG; i = 1)
    N, B = length(source), capacity(target)
    ptot = prefixsum(source.priorities, N)
    Δp = ptot / B
    ids = [inverse_query(source.priorities, (j + rand(rng) - 1) * Δp) for j=1:B]
    pmin = first(source.minsort_priorities) / ptot
    max_w = (pmin*N)^(-source.β(i))
    ws = [(N * source.priorities[id] / ptot)^source.β(i) for id in ids] ./ max_w
    
    # Add the indices to the target and the weights to the target
    target.indices = ids
    for k in keys(target.data)
        k == :weight && continue
        copyto!(target.data[k], source[k][:, ids])
    end
    !haskey(target.data, :weight) && (target.data[:weight] = deepcopy(target.data[:r]))
    copyto!(target.data[:weight], ws)
    target.elements = B
end
