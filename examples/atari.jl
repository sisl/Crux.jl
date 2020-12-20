using POMDPs, POMDPPolicies, POMDPGym, Crux

mdp = GymPOMDP(:PongNoFrameskip, version = :v4)

s = rand(initialstate(mdp))
torgb(s)
sp, r, o = gen(mdp, s, 1)
actions(mdp)

as = actions(mdp)
s_dim = size(s)
a_dim = length(as)

sampler = Sampler(mdp, RandomPolicy(mdp), s_dim, a_dim)

episodes!(sampler)

