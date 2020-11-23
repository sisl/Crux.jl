@with_kw mutable struct ExperienceBuffer{T <: AbstractArray} 
    data::Dict{Symbol, T}
    capacity::Int64
    next_ind::Int64 = 1
    
    indices::Vector{Int64}
    priorities::MutableBinaryHeap{Float64, DataStructures.FasterForward}
    priority_sums::FenwickTree
end

mdp_data(sdim::Int, adim::Int, size::Int; Atype = Array{Float32,2}) = 
        Dict(
            :s => Atype(undef, sdim, size), 
            :a => Atype(undef, adim, size), 
            :sp => Atype(undef, sdim, size), 
            :r => Atype(undef, 1, size), 
            :done => Atype(undef, 1, size)
            )
    
function ExperienceBuffer(sdim::Int, adim::Int, capacity::Int; device = cpu, gae = false, Ninit = 0)
    Atype = device == gpu ? CuArray{Float32,2} : Array{Float32,2}
    data = mdp_data(sdim, adim, capacity, Atype = Atype)
    if gae
        b.data[:return] = Atype(undef, 1, size)
        b.data[:advantage] = Atype(undef, 1, size)
    end
    ExperienceBuffer(data = data, capacity = capacity)
end

#TODO Change all instances of "size" to capacity
# ExperienceBuffer(mdp, capacity::Int; device = cpu, gae = false, Ninit = 0) = ExperienceBuffer(sdim(mdp), adim(mdp), capacity, device = device, gae = gae, Ninit = Ninit)

function ExperienceBuffer(b::ExperienceBuffer; device = device(b))
    data = Dict(k => todevice(v, device) for (k,v) in b.data)
    ExperienceBuffer(data, b.capacity, b.next_ind, b.indices, b.priorities, b.priority_sums)
end

#TODO Implement everything to treat experience buffer as a dictionary
Base.getindex(b::ExperienceBuffer, key::Symbol) = view(b.data[key], :, 1:b.elements)

Base.keys(b::ExperienceBuffer) = keys(b.data)

Base.length(b::ExperienceBuffer) = b.elements

capacity(b::ExperienceBuffer) = size(first(b.data)[2], 2)

clear!(b::ExperienceBuffer) = b.elements, b.next_ind = 0,1

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
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
end


function Base.fill!(b_in::ExperienceBuffer, mdp, N = capacity(b_in); policy = RandomPolicy(mdp), rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100)
    b = isgpu(b_in) ? empty_like(b_in, device = cpu) : b_in
    clear!(b)
    push_episodes!(b, mdp, N, policy = policy, rng = rng, baseline = baseline, max_steps = max_steps)
    isgpu(b_in) && copyto!(b_in, b) # Copy from cpu buffer to the given gpu buffer
end

function Base.fill(::Type{ExperienceBuffer}, mdp, N::Int; policy = RandomPolicy(mdp), capacity = N, rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100, device = cpu)
    buffer = ExperienceBuffer(mdp, capacity, device = device)
    fill!(buffer, mdp, N, policy = policy, rng = rng, baseline = baseline, max_steps = max_steps)
    buffer
end

function Base.merge(buffers::ExperienceBuffer...; capacity = sum(length.(buffers)))
    new_buff = empty_like(buffers[1], capacity = capacity)
    for b in buffers
        push!(new_buff, b)
    end
    new_buff            
end
    
function Random.rand!(rng::AbstractRNG, target::ExperienceBuffer, source::ExperienceBuffer)
    @assert length(target) <= length(source)
    N = length(target)
    ids = rand(rng, 1:source.elements, N)
    for k in keys(target.data)
        copyto!(target[k], source[k][:, ids])
    end
end



