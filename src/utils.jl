sdim(mdp) = length(convert_s(AbstractVector, rand(initialstate(mdp)), mdp))
adim(mdp) = length(actions(mdp))

## GPU Stuff
function Base.copyto!(Cto::Chain, Cfrom::Chain)
    for i = 1:length(Flux.params(Cto).order.data)
        copyto!(Flux.params(Cto).order.data[i], Flux.params(Cfrom).order.data[i])
    end
end

todevice(C, device) = (device == gpu) ? (C |> gpu) : nothing

device(v::AbstractArray) = (v isa CuArray) ? gpu : cpu

function LinearAlgebra.norm(grads, params, p::Real = 2)
    norms = []
    for param in params
        push!(norm(grads[param].data[:], p))
    end
    norm(norms, p)
end




