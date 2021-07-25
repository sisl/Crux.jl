## Spaces
abstract type AbstractSpace end
@with_kw struct DiscreteSpace <: AbstractSpace
    N::Int
    vals = collect(1:N)
end
DiscreteSpace(N::Int) = DiscreteSpace(N = N)
DiscreteSpace(vals::A) where {A <: AbstractArray} = DiscreteSpace(length(vals), vals)
DiscreteSpace(vals::A) where {A <: Tuple} = DiscreteSpace(length(vals), vals)

@with_kw struct ContinuousSpace{T <: Tuple} <: AbstractSpace 
    dims::T
    type::Type = Float32
    μ = 0f0
    σ = 1f0
end
ContinuousSpace(dims, type::Type=Float32; kwargs...) = ContinuousSpace(dims = tuple(dims...), type=type; kwargs...)

type(S::DiscreteSpace) = Bool
type(S::ContinuousSpace) = S.type

dim(S::DiscreteSpace) = (S.N,)
dim(S::ContinuousSpace) = S.dims

tovec(v, S::DiscreteSpace) = Flux.onehot(v, S.vals)
tovec(v, S::ContinuousSpace) = whiten(v, S.μ, S.σ)

function state_space(o::AbstractArray; μ=0f0, σ=1f0)
    dims = size(o)
    length(dims) == 4 && dims[end] == 1 && (dims = dims[1:end-1]) # Handle image-like inputs that need to be 4D for conv layers
    ContinuousSpace(dims, typeof(o[1]), μ, σ)
end

function state_space(mdp; μ=0f0, σ=1f0)
    if mdp isa MDP
        o = convert_s(AbstractArray, rand(initialstate(mdp)), mdp)
    elseif mdp isa POMDP
        s = rand(initialstate(mdp))
        o = rand(initialobs(mdp, s))
    else
        error("Unrecognized problem: ", mdp)
    end
    return state_space(o, μ=μ, σ=σ)
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

whiten(v) = whiten(v, mean(v), std(v))
whiten(v, μ, σ) = (v .- μ) ./ σ

to2D(W) = reshape(W, :, size(W, ndims(W))) # convert a multidimensional weight matrix to 2D

# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)
    
function LinearAlgebra.norm(grads::Flux.Zygote.Grads; p::Real = 2)
    v = []
    for θ in grads.params
        !isnothing(grads[θ]) && push!(v, norm(grads[θ] |> cpu, p))
    end
    norm(v, p)
end

