@with_kw mutable struct BatchAdjacentBuffer{T1, T2}
    batches::Vector{T1} = []
    z_dists::Vector{T2} = []
    elements::Int = 0 # max capacity
    next_ind::Int = 1 # Location of the next index for the circular buffer
    total_count::Int = 0
    default_z_dist
    reference_buffer
end

isprioritized(b::BatchAdjacentBuffer) = false

extra_columns(b::BatchAdjacentBuffer) = extra_columns(b.reference_buffer)

function fillto!(b::BatchAdjacentBuffer, s::Union{Sampler, Vector{T}}, N::Int; i=1, explore=false) where {T <: Sampler}
    Nfill = max(0, N - length(b))
    
    if Nfill > 0
        D = steps!(s, i=i, Nsteps=Nfill, explore=explore)
        D[:z] = repeat(b.default_z_dist.μ, 1, length(D[:r]))
        push!(b, D, deepcopy(b.default_z_dist))
    end
    Nfill
end

DataStructures.capacity(b::BatchAdjacentBuffer) = b.elements

Base.length(buffer::BatchAdjacentBuffer) = length(buffer.batches) > 0 ? sum([length(b) for b in buffer.batches]) : 0

function device(b::BatchAdjacentBuffer)
    d = device(b.batches[1])
    for e in b.batches
        @assert device(e) == d
    end
    return d
end

function buffer_like(b::BatchAdjacentBuffer; capacity=capacity(b.reference_buffer[1]), device=device(b.reference_buffer))
    buffer_like(b.reference_buffer, capacity=capacity, device=device)
end

Base.push!(b::BatchAdjacentBuffer, data) = nothing #push!(b, data, nothing)

function Base.push!(b::BatchAdjacentBuffer, data, z_dist)
    # Convert data to an experience buffer if needed
    if !(data isa ExperienceBuffer)
        data = ExperienceBuffer(data)
    end
    
    # Add the new buffer to the array
    if length(b.batches) < capacity(b)
        push!(b.batches, data)
        push!(b.z_dists, z_dist)
    else
        b.batches[b.next_ind] = data
        b.z_dists[b.next_ind] = z_dist
    end
    
    b.next_ind = mod1(b.next_ind + 1, b.elements)
end


function push_reservoir!(b::BatchAdjacentBuffer, data, z_dist; weight=nothing)
    # Convert data to an experience buffer if needed
    if !(data isa ExperienceBuffer)
        data = ExperienceBuffer(data)
    end
    
    # Only add samples according to their weight
    if !isnothing(weight) && rand() > weight
        return
    end
    
    b.total_count += 1
    
    # Add the element to the buffer and return if we haven't hit our max
    if length(b.batches) < capacity(b)
        push!(b.batches, data)
        push!(b.z_dists, z_dist)
    else
        # Choose a random number up to count
        j = rand(1:b.total_count)
        
        # If its within the buffer replace the element
        if j <= capacity(b)
            b.batches[j] = data
            b.z_dists[j] = z_dist
        end
    end
end


function Random.rand!(target::ExperienceBuffer, source::BatchAdjacentBuffer...; i = 1, fracs = ones(length(source))./length(source))
    lens = [length.(source) ...]
    if any(lens .== 0)
        fracs[lens .== 0] .= 0
        fracs ./= sum(fracs)
    end
    batches = split_batches(capacity(target), fracs)
    for (b, B) in zip(source, batches)
        B == 0 && continue
        for i=1:B
            batch_sizes = length.(b.batches)
            d = Categorical(batch_sizes ./ sum(batch_sizes))
            i1 = rand(d)
            i2 = rand(1:length(b.batches[i1]))
            push!(target, b.batches[i1], ids = [i2])
        end
    end
end


function log_episode_averages(buffer::BatchAdjacentBuffer, keys, period)
    (;kwargs...) -> begin
        d = Dict()
        indices = get_last_N_indices(buffer, period)
        for k in keys
            val = 0
            ends = 0
            for i in indices
                val += sum(buffer.batches[i][k])
                ends += sum(buffer.batches[i][:episode_end])
            end
            d[Symbol(string("avg_", k))] = val / ends
        end
        d
    end
end

function get_last_N_indices(b::BatchAdjacentBuffer, N)
    N = min(length(b.batches), N)
    C = capacity(b)
    start_index = mod1(b.next_ind - N, C)
    mod1.(start_index:start_index + N - 1, C)
end

