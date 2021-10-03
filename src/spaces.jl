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