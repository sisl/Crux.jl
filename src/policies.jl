## Policy Representations
mutable struct CategoricalPolicy <: Policy
    Q
    mdp
    Qgpu
end

CategoricalPolicy(Q, mdp; device = cpu) = CategoricalPolicy(Q, mdp, device == gpu ? Q |> gpu : nothing)

Flux.params(p::CategoricalPolicy, device) = (device == gpu) ? Flux.params(p.Qgpu) : Flux.params(p.Q)

network(p::CategoricalPolicy, device) = (device == gpu) ? p.Qgpu : p.Q

sync!(p::CategoricalPolicy, device) = (device == gpu) ? copyto!(p.Q, p.Qgpu) : nothing

POMDPs.action(p::CategoricalPolicy, s) = actions(mdp)[argmax(p.Q(convert_s(AbstractVector, s, p.mdp)))]

#logprob
#entropy
#kldivergence



struct GaussianPolicy
    
end