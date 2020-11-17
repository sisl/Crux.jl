include("../src/includes.jl") # this will be replaced with a using statement eventually
include("mdps/lavaworld.jl")


# Setup the problem parameters
sz = (7,5)
lava = [(3,1), (4,1), (5,1), (3,5), (4,5), (5,5)]
# lava2 = [(3,4), (4,4), (5,4), (3,5), (4,5), (5,5)]
lava_penalty = -1.0
goals = [(7,5)]
goal_reward = 1.0
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
Qnet(args...) = Chain(Dense(input_dim, 256, relu), Dense(256,64, relu), Dense(64, 4), args...) 
dqn_steps = 30000 # to learn an expert policy
gail_steps = 10000
expert_buffer_size = 1000 
nda_buffer_size = 1000
λ_nda = 0.5f0 # Constant for NDA. λ = 1 ignores the NDA trajectories
# λ_ent = 0.001f0


# Build the mdp
# mdp = SimpleGridWorld(size = sz, tprob = 1.)
mdp = SimpleGridWorld(size = sz, tprob = 1., rewards = lavaworld_rewards(lava, lava_penalty, goals, goal_reward))
# mdp2 = SimpleGridWorld(size = sz, tprob = 1., rewards = lavaworld_rewards(lava2, lava_penalty, goals, goal_reward))

## solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Qnet(), mdp),
                  N=dqn_steps, 
                  batch_size = 128, 
                  exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=dqn_steps/2))
                  )
π_dqn = solve(𝒮_dqn, mdp)

## Fill a buffer with expert trajectories
expert_trajectories = ExperienceBuffer(mdp, 1000)
fill!(expert_trajectories, mdp, π_dqn)

# Solve with GAIL
𝒮_gail = GAILSolver(π = DQNPolicy(Qnet(), mdp), 
                    D = DQNPolicy(Qnet(softmax), mdp),
                    N = gail_steps,
                    opt = ADAM(1e-4),
                    expert_buffer = expert_trajectories,
                    batch_size = 128,
                    exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=gail_steps/2))
                    )
π_gail = solve(𝒮_gail, mdp)

# Solve with NDA-GAIL
# nda_trajectories = gen_buffer(mdp, RandomPolicy(mdp), 10000, desired_return = -1., nonzero_transitions_only = true)
# nda_gail_net = train_GAIL!(mdp, Qnet(), Dnet(), expert_trajectories, 
#                               nda_buff = nda_trajectories,  
#                               logdir = "log/nda-gail/", 
#                               λ_nda = λ_nda, 
#                               epochs = gail_steps)

## Make some plots
using Cairo, Fontconfig, Compose, ColorSchemes
set_default_graphic_size(35cm,10cm)
r = compose(Compose.context(0,0,1cm, 0cm), Compose.rectangle()) # spacer

# Plot on the training MDP
expert_occupancy = gen_occupancy(expert_trajectories, mdp)
c_expert = render(mdp, (s = GWPos(7,5),), color = s->reward(mdp,s) <0 ? -10. :  Float64(expert_occupancy[s]) / 2., policy = π_dqn)
c_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy =π_gail)
c_nda_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy = ChainPolicy(nda_gail_net,mdp))
hstack(c_expert, r, c_gail, r, c_nda_gail) |> SVG("images/mdp1.svg")

c_expert2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(dqn_net, mdp2))
c_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(gail_net, mdp2))
c_nda_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(nda_gail_net, mdp2))
hstack(c_expert2, r, c_gail2, r, c_nda_gail2) |> SVG("images/mdp2.svg")

nda_occupancy = gen_occupancy(nda_trajectories, mdp2)
c_expert = render(mdp2, (s = GWPos(7,5),), color = s-> Float64(nda_occupancy[s]) / 1.6, policy = ChainPolicy(dqn_net, mdp))


