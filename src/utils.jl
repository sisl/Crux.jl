## Spaces
abstract type AbstractSpace end
@with_kw struct DiscreteSpace <: AbstractSpace
    N::Int
    onehot::Bool = true
    type::Type = Bool
end
DiscreteSpace(N::Int) = DiscreteSpace(N = N)

@with_kw struct ContinuousSpace{T <: Tuple} <: AbstractSpace 
    dims::T
    type::Type = Float32
end
ContinuousSpace(dims) = ContinuousSpace(dims = tuple(dims...))

useonehot(s::DiscreteSpace) = s.onehot
type(s::DiscreteSpace) = s.onehot ? Bool : s.type
dim(s::DiscreteSpace) = s.onehot ? (s.N,) : (1,)

useonehot(s::ContinuousSpace) = false
type(s::ContinuousSpace) = s.type
dim(s::ContinuousSpace) = s.dims

function state_space(mdp)
    if mdp isa MDP
        o = convert_s(AbstractArray, rand(initialstate(mdp)), mdp)
    else
        s = rand(initialstate(mdp))
        o = rand(initialobs(mdp, s))
    end
    dims = o isa AbstractArray ? size(o) : (1,)
    length(dims) == 4 && dims[end] == 1 && (dims = dims[1:end-1]) # Handle image-like inputs that need to be 4D for conv layers
    ContinuousSpace(dims, typeof(o[1]))
end

## GPU Stuff
device(v::T) where T <: CuArray = gpu
device(v::T) where T <: AbstractArray = cpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = cpu
function device(c) 
    p = Flux.params(c)
    length(p) > 0 && p[1] isa CuArray ? gpu : cpu
end 

function Base.copyto!(Cto::Chain, Cfrom::Chain)
    for i = 1:length(Flux.params(Cto).order.data)
        copyto!(Flux.params(Cto).order.data[i], Flux.params(Cfrom).order.data[i])
    end
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

## Flux stuff
function LinearAlgebra.norm(grads::Flux.Zygote.Grads; p::Real = 2)
    v = []
    for θ in grads.params
        !isnothing(grads[θ]) && push!(v, norm(grads[θ] |> cpu, p))
    end
    norm(v, p)
end

