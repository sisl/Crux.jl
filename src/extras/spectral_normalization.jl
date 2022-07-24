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
struct DenseSN{F,S<:AbstractMatrix, B, I<:Int, VV<:AbstractArray}
  weight::S
  bias::B
  σ::F
  n_iterations::I # Number of power iterations for computing max singular value
  u::VV # Left vector for power iteration
  function DenseSN(W::M, bias=true, σ::F=identity, n_iterations::I=1, u::VV=randn(Float32, size(W,1), 1)) where {M<:AbstractMatrix, F, I<:Int, VV<:AbstractArray}
    b = Flux.create_bias(W, bias, size(W,1))
    new{F,M,typeof(b), I, VV}(W, b, σ, n_iterations, u)
  end
end

function DenseSN(in::Integer, out::Integer, σ=identity; init=Flux.glorot_uniform, bias=true, n_iterations=1, u=randn(Float32, out, 1))
  DenseSN(init(out, in), bias, σ, n_iterations, u)
end

Flux.@functor DenseSN

Flux.trainable(a::DenseSN) = (a.weight, a.bias)

function (a::DenseSN)(x::AbstractVecOrMat)
  W, b, σ = a.weight, a.bias, a.σ
  u, v = ignore_derivatives(() -> power_iteration!(W, a.u, a.n_iterations))
  σ.((W ./ msv(u, v, W))*x .+ b)
end

(a::DenseSN)(x::AbstractArray) = reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)

function Base.show(io::IO, l::DenseSN)
  print(io, "DenseSN(", size(l.weight, 2), ", ", size(l.weight, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
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
  u, v = ignore_derivatives(() -> power_iteration!(to2D(c.weight), c.u, c.n_iterations))
  σ.(conv(x, c.weight ./ msv(u, v, to2D(c.weight)), cdims) .+ b)
end

function Base.show(io::IO, l::ConvSN)
  print(io, "ConvSN(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end