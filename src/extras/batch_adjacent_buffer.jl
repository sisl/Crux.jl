struct BatchAdjacentBuffer{T1, T2}
    batches::Vector{T1}
    z_dists::Vector{T2}
    elements::Int # max capacity
end

function device(b::BatchAdjacentBuffer)
    d = device(b.batches[1])
    for e in b.batches
        @assert device(e) == d
    end
    return d
end

function buffer_like(b::BatchAdjacentBuffer; capacity=capacity(b), device=device(b))
    data = Dict(k=>device(Array{eltype(v)}(undef, size(v)[1:end-1]..., capacity)) for (k,v) in b.data)
    ExperienceBuffer(data, 0, 1, Int[], isprioritized(b) ? PriorityParams(capacity, b.priority_params) : nothing, 0)
end


# Note: data can be a dictionary or an experience buffer
function Base.push!(b::ExperienceBuffer, data; ids = nothing)
    ids = isnothing(ids) ? UnitRange(1, size(data[first(keys(data))], 2)) : ids
    N, C = length(ids), capacity(b)
    b.total_count += N
    I = mod1.(b.next_ind:b.next_ind + N - 1, C)
    for k in keys(b)
        if !haskey(data, k)
            # CRUX_WARNINGS && @warn "Pushed data does not contain $k"
            continue
        end
        v1 = bslice(b.data[k], I)
        v2 = collect(bslice(data[k], ids))
        @assert size(v1)[1:end-1] == size(v2)[1:end-1]
        copyto!(v1, v2)
    end
    isprioritized(b) && update_priorities!(b, I, b.priority_params.max_priority*ones(N))
        
    b.elements = min(C, b.elements + N)
    b.next_ind = mod1(b.next_ind + N, C)
    I
end


function push_reservoir!(buffer, data; weighted=false)
    N = capacity(data)
    for i=1:N
        element = Dict(k => bslice(v, i:i) for (k,v) in data)
        
        # Only add samples according to their weight
        if weighted && haskey(element, :weight) && rand() > element[:weight][1]
            continue
        end
        
        buffer.total_count += 1
        # Add the element to the buffer and return if we haven't hit our max
        if length(buffer) < capacity(buffer)
            push!(buffer, element)
        else
            # Choose a random number up to count
            j = rand(1:buffer.total_count)
            
            # If its within the buffer replace the element
            if j <= capacity(buffer)
                for k in keys(buffer)
                    bslice(buffer[k], j:j) .= element[k]
                end
            end
        end
    end
end


function Random.rand!(target::ExperienceBuffer, source::ExperienceBuffer...; i = 1, fracs = ones(length(source))./length(source))
    lens = [length.(source) ...]
    if any(lens .== 0)
        fracs[lens .== 0] .= 0
        fracs ./= sum(fracs)
    end
    batches = split_batches(capacity(target), fracs)
    for (b, B) in zip(source, batches)
        B == 0 && continue
        isprioritized(b) ? prioritized_sample!(target, b, i=i, B=B) : uniform_sample!(target, b, B=B)
    end
end