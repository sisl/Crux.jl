using POMDPModels, POMDPModelTools

POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(transition(mdp, s, a )), r = reward(mdp, s, a))

function POMDPs.initialstate(mdp::SimpleGridWorld)
    while true
        x, y = rand(1:mdp.size[1]), rand(1:mdp.size[2])
        !(GWPos(x,y) in mdp.terminate_from) && return Deterministic(GWPos(x,y))
    end
end 

POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, mdp::SimpleGridWorld) = Float32.([s...])