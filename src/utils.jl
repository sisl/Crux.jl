## MDP/POMDP helpers
sdim(mdp) = length(convert_s(AbstractVector, rand(initialstate(mdp)), mdp))
adim(mdp) = length(actions(mdp))

## GPU Stuff
device(v::T) where T <: CuArray = gpu
device(v::T) where T <: AbstractArray = cpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: CuArray, I, L} = gpu
device(v::SubArray{T,N,P,I,L}) where {T, N, P <: AbstractArray, I, L} = cpu

todevice(C, device) = (device == gpu) ? (C |> gpu) : nothing

function Base.copyto!(Cto::Chain, Cfrom::Chain)
    for i = 1:length(Flux.params(Cto).order.data)
        copyto!(Flux.params(Cto).order.data[i], Flux.params(Cfrom).order.data[i])
    end
end

bslice(v, i) = view(v, ntuple(x->:, ndims(v)-1)..., i)

## Flux stuff
LinearAlgebra.norm(grads::Flux.Zygote.Grads; p::Real = 2) = norm([norm(grads[θ] |> cpu, p) for θ in grads.params], p)
