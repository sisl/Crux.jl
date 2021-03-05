# Power iteration algorithm for computing the spectral norm
function power_iteration!(W, u, n_iterations)
    v = nothing
    for i=1:n_iterations
        Wu = W' * u
        v = Wu ./ (norm(Wu) + eps(Float32))
        Wv = W * v
        u .= Wv ./ (norm(Wv) + eps(Float32))
    end
    u, v
end

# Compute the maximum singular value
msv(u, v, W) = u' * W * v


## Dense Layer with Spectral Normalization
struct DenseSN{F,S<:AbstractArray,T<:AbstractArray, I<:Int, VV<:AbstractArray}
  W::S
  b::T
  σ::F
  n_iterations::I # Number of power iterations for computing max singular value
  u::VV # Left vector for power iteration
end

function DenseSN(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = Flux.zeros, n_iterations = 1)
  return DenseSN(initW(out, in), initb(out), σ, n_iterations, randn(Float32, out, 1))
end

Flux.@functor DenseSN

Flux.trainable(a::DenseSN) = (a.W, a.b)

function (a::DenseSN)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  u, v = Zygote.ignore(() -> power_iteration!(W, a.u, a.n_iterations))
  σ.((W ./ msv(u, v, W))*x .+ b)
end

function Base.show(io::IO, l::DenseSN)
  print(io, "DenseSN(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::DenseSN{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::DenseSN{<:Any,W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

function outdims(l::DenseSN, isize)
    first(isize) == size(l.W, 2) || throw(DimensionMismatch("input size should equal to ($(size(l.W, 2)),), got $isize"))
    return (size(l.W, 1),)
end


## Convluational layer with Spectral Normalization
struct ConvSN{N,M,F,A,V, I<:Int, VV<:AbstractArray}
  σ::F
  weight::A
  bias::V
  stride::NTuple{N,Int}
  pad::NTuple{M,Int}
  dilation::NTuple{N,Int}
  n_iterations::I # Number of power iterations for computing max singular value
  u::VV # Left vector for power iteration
end

to2D(W) = reshape(W, :, size(W, ndims(W)))

function ConvSN(w::AbstractArray{T,N}, b::Union{Flux.Zeros, AbstractVector{T}}, σ = identity;
              stride = 1, pad = 0, dilation = 1, n_iterations = 1) where {T,N}
  stride = Flux.expand(Val(N-2), stride)
  dilation = Flux.expand(Val(N-2), dilation)
  pad = Flux.calc_padding(Conv, pad, size(w)[1:N-2], dilation, stride)
  u = randn(Float32, size(to2D(w), 1), 1)
  return ConvSN(σ, w, b, stride, pad, dilation, n_iterations, u)
end

function ConvSN(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity;
            init = Flux.glorot_uniform,  stride = 1, pad = 0, dilation = 1,
            weight = Flux.convfilter(k, ch, init = init), bias = Flux.zeros(ch[2]), n_iterations = 1) where N
  ConvSN(weight, bias, σ, stride = stride, pad = pad, dilation = dilation, n_iterations = n_iterations)
end

Flux.@functor ConvSN

Flux.trainable(a::ConvSN) = (a.weight, a.bias)

function (c::ConvSN)(x::AbstractArray)
  σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
  cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
  u, v = Zygote.ignore(() -> power_iteration!(to2D(c.weight), c.u, c.n_iterations))
  σ.(conv(x, c.weight ./ msv(u, v, to2D(c.weight)), cdims) .+ b)
end

function Base.show(io::IO, l::ConvSN)
  print(io, "ConvSN(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end

(a::ConvSN{<:Any,<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::ConvSN{<:Any,<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))


outdims(l::ConvSN, isize) =
  output_size(DenseConvDims(_paddims(isize, size(l.weight)), size(l.weight); stride = l.stride, padding = l.pad, dilation = l.dilation))

