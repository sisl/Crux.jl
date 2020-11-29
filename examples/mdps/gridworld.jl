using POMDPs, POMDPModels, POMDPModelTools, Random

POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(rng, transition(mdp, s, a )), r = reward(mdp, s, a))

function POMDPs.initialstate(mdp::SimpleGridWorld)
    function istate(rng::AbstractRNG)
        while true
            x, y = rand(rng, 1:mdp.size[1]), rand(rng, 1:mdp.size[2])
            !(GWPos(x,y) in mdp.terminate_from) && return GWPos(x,y)
        end
    end
    return ImplicitDistribution(istate)
end 

POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, mdp::SimpleGridWorld) = Float32.([s...])