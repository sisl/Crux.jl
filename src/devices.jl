device(v::T) where T <: CuArray = gpu
device(v::T) where T <: AbstractArray = cpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = cpu
function device(c) 
    p = Flux.params(c)
    length(p) > 0 && p[1] isa CuArray ? gpu : cpu
end 

# Call F with input x but ensure they are both on the device of F
gpucall(F, x::CuArray) = F(x)
gpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = F(x)

gpucall(F, x::Array) = cpu(F(gpu(x)))
gpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = cpu(F(gpu(collect(x))))

cpucall(F, x::Array) = F(x)
cpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = F(x)

cpucall(F, x::CuArray) = gpu(F(cpu(x)))
cpucall(F, x::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu(F(cpu(x)))

mdcall(F, x, device) = device == gpu ? gpucall(F,x) : cpucall(F, x)

@inline function bslice(v, i)
    nd = ndims(v)
    if nd == 2
        return view(v,:,i)
    elseif nd == 3
        return view(v, :, :, i)
    elseif nd == 4
        return view(v, :, :, :, i)
    else
        return view(v, ntuple(x->:, nd-1)..., i)
    end
end