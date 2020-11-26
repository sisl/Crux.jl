## MDP/POMDP helpers
sdim(mdp) = length(convert_s(AbstractVector, rand(initialstate(mdp)), mdp))
adim(mdp) = length(actions(mdp))

## Efficient inverse query for fenwick tree 
# Taken from https://codeforces.com/blog/entry/61364
function inverse_query(t::FenwickTree, v)
    tot, pos, N = 0, 0, length(t)
    for i=floor(Int, log2(N)):-1:0
        new_pos = pos + 1 << i
        if new_pos <= N && tot + t.bi_tree[new_pos] < v
            tot += t.bi_tree[new_pos]
            pos = new_pos
        end
    end
    pos + 1
end

Base.getindex(t::FenwickTree, i::Int) = prefixsum(t, i) - prefixsum(t, i-1)

DataStructures.update!(t::FenwickTree, i, v) = inc!(t, i, v - t[i])

## GPU Stuff
device(v::CuArray) = gpu
device(v::AbstractArray) = cpu

todevice(C, device) = (device == gpu) ? (C |> gpu) : nothing

function Base.copyto!(Cto::Chain, Cfrom::Chain)
    for i = 1:length(Flux.params(Cto).order.data)
        copyto!(Flux.params(Cto).order.data[i], Flux.params(Cfrom).order.data[i])
    end
end

## Flux stuff
LinearAlgebra.norm(grads::Flux.Zygote.Grads, p::Real = 2) = norm([norm(grads[θ] |> cpu, p) for θ in grads.params], p)

