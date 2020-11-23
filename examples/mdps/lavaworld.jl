using POMDPs, POMDPModels, POMDPModelTools, Random
using Cairo, Fontconfig, Compose, ColorSchemes

POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(transition(mdp, s, a )), r = reward(mdp, s, a))

lavaworld_rewards(lava, lava_penalty, goals, goal_reward) = merge(Dict(GWPos(p...) => lava_penalty for p in lava), Dict(GWPos(p...) => goal_reward for p in goals))

function random_lava(size, num_tiles; goal = :random, lava_penalty = -1, goal_reward = 1, rng = Random.GLOBAL_RNG)
    g = goal == :random ? (rand(rng, 1:size[1]), rand(rng, 1:size[2])) :  goal
    lava = []
    while length(lava) < num_tiles
        p = (rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        p != g && push!(lava, p)
    end 
    @assert !(g in lava)
    lavaworld_rewards(lava, lava_penalty, [g], goal_reward)
end

function update_lava!(mdp::SimpleGridWorld)
    [delete!(mdp.rewards, k) for k in keys(mdp.rewards)]
    [delete!(mdp.terminate_from, k) for k in mdp.terminate_from]
    rs = random_lava(mdp.size, 6)
    for k in keys(rs)
        mdp.rewards[k] = rs[k]
        push!(mdp.terminate_from, k)
    end
end

function POMDPs.initialstate(mdp::SimpleGridWorld)
    # update_lava!(mdp)
    # return Deterministic(GWPos(1,5))
    while true
        x, y = rand(1:mdp.size[1]), rand(1:mdp.size[2])
        !(GWPos(x,y) in mdp.terminate_from) && return Deterministic(GWPos(x,y))
    end
end 
            
function POMDPs.convert_s(::Type{V}, s::GWPos, mdp::SimpleGridWorld) where {V<:AbstractArray}
    svec = zeros(Float32, mdp.size..., 3)
    !isterminal(mdp, s) && (svec[s[1], s[2], 3] = 1.)
    for p in states(mdp)
        reward(mdp, p) < 0 && (svec[p[1], p[2], 2] = 1.)
        reward(mdp, p) > 0 && (svec[p[1], p[2], 1] = 1.)
    end
    svec[:]
end

POMDPs.convert_s(::Type{GWPos}, v::V, mdp::SimpleGridWorld) where {V<:AbstractArray} = GWPos(findfirst(reshape(v, mdp.size..., :)[:,:,3] .== 1.0).I)

goal(mdp, s) = GWPos(findfirst(reshape(s, mdp.size..., :)[:,:,1] .== 1.0).I)

function gen_occupancy(buffer, mdp)
    occupancy = Dict(s => 0 for s in states(mdp))
    for i=1:length(buffer)
        s = convert_s(GWPos, buffer[:s][:,i], mdp)
        occupancy[s] += 1
    end
    occupancy
end

simple_display(mdp::SimpleGridWorld, color = s->10.0*reward(mdp, s), policy= nothing, s0 = GWPos(7,5)) = render(mdp, (s = s0,), color = color, policy = policy)

render_and_save(filename, g::MDP...) = hcat_and_save(filename,  [simple_display(gi) for gi in g]...)

function hcat_and_save(filename, c::Context...)
    set_default_graphic_size(35cm,10cm)
    r = compose(Compose.context(0,0,1cm, 0cm), Compose.rectangle()) # spacer
    cs = []
    for ci in c
        push!(cs, ci)
        push!(cs, r)
    end
    hstack(cs[1:end-1]...) |> PDF(filename)
end

