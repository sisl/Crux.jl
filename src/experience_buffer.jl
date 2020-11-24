const MinHeap = MutableBinaryHeap{Float32, DataStructures.FasterForward}

@with_kw mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    capacity::Int64
    next_ind::Int64 = 1
    
    indices::Vector{Int} = []
    priorities::Union{Nothing, MinHeap} = nothing
    priority_sums::Union{Nothing, FenwickTree} = nothing
end

function mdp_data(sdim::Int, adim::Int, size::Int; Atype = Array{Float32,2}, gae = false)
    data = Dict(
        :s => Atype(undef, sdim, size), 
        :a => Atype(undef, adim, size), 
        :sp => Atype(undef, sdim, size), 
        :r => Atype(undef, 1, size), 
        :done => Atype(undef, 1, size)
        )
    if gae
        b.data[:return] = Atype(undef, 1, size)
        b.data[:advantage] = Atype(undef, 1, size)
    end
end
    
function ExperienceBuffer(sdim::Int, adim::Int, capacity::Int; device = cpu, gae = false, Ninit = 0, prioritized = false)
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

isgpu(b::ExperienceBuffer{CuArray{Float32, 2}}) = true
isgpu(b::ExperienceBuffer{Array{Float32, 2}}) = false

# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data)
    N, C = size(data[first(keys(data))], 2), capacity(b)
    I = circular_indices(b.next_ind, N, C)
    for k in keys(data)
        b.data[k][:, I] .= data[k]
    end
    prioritized(b) && update_priorties!(b, I, MAX_PRIORITY*ones(N))
        
        
    end
        
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
end

function update_priorities!(b, I::AbstractArray, v::AbstractArray)
    for i=1:I
        update!(b.priorities, i, v[i])
        update!(b.priority_sums, i, v[i])
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


# function Base.fill!(b_in::ExperienceBuffer, mdp, N = capacity(b_in); policy = RandomPolicy(mdp), rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100)
#     b = isgpu(b_in) ? empty_like(b_in, device = cpu) : b_in
#     clear!(b)
#     push_episodes!(b, mdp, N, policy = policy, rng = rng, baseline = baseline, max_steps = max_steps)
#     isgpu(b_in) && copyto!(b_in, b) # Copy from cpu buffer to the given gpu buffer
# end
# 
# function Base.fill(::Type{ExperienceBuffer}, mdp, N::Int; policy = RandomPolicy(mdp), capacity = N, rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100, device = cpu)
#     buffer = ExperienceBuffer(mdp, capacity, device = device)
#     fill!(buffer, mdp, N, policy = policy, rng = rng, baseline = baseline, max_steps = max_steps)
#     buffer
# end
# 
# function Base.merge(buffers::ExperienceBuffer...; capacity = sum(length.(buffers)))
#     new_buff = empty_like(buffers[1], capacity = capacity)
#     for b in buffers
#         push!(new_buff, b)
#     end
#     new_buff            
# end
