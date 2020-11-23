## MDP/POMDP helpers
sdim(mdp) = length(convert_s(AbstractVector, rand(initialstate(mdp)), mdp))
adim(mdp) = length(actions(mdp))

## Efficient inverse query for fenwick tree 
# Taken from https://codeforces.com/blog/entry/61364
function inv_query(t::FenwickTree, v)
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

value(t::FenwickTree, i) = t[i] - t[i-1]
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




