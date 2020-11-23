include("../src/includes.jl") # this will be replaced with a using statement eventually
include("mdps/lavaworld.jl")


# Setup the problem parameters
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
N_nda = 100
N_test = 100
expert_task = SimpleGridWorld(size = sz, tprob = 1.0, rewards = random_lava(sz, 1, goal = (7,5), rng = MersenneTwister(0)))
nda_tasks = [SimpleGridWorld(size = sz, tprob = 1.0,  rewards = random_lava(sz, 1, goal = (7,5))) for _=1:N_nda]
test_tasks = [SimpleGridWorld(size = sz, tprob = 1.0,  rewards = random_lava(sz, 1, goal = (7,5))) for _=1:N_test]

simple_display(expert_task)
simple_display(nda_tasks[3])
simple_display(test_tasks[9])

Qnet(args...) = Chain(Dense(input_dim, 128, relu), Dense(128,64, relu), Dense(64, 4), args...) 

dqn_steps = 20000 # to learn an expert policy
gail_steps = 10000
expert_buffer_size = 1000 
nda_buffer_size = 1000
λ_nda = 0.5f0 # Constant for NDA. λ = 1 ignores the NDA trajectories

## solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Qnet(), expert_task, device = gpu),
                  N=dqn_steps, 
                  batch_size = 128, 
                  exploration_policy = EpsGreedyPolicy(expert_task, LinearDecaySchedule(start=1., stop=0.1, steps=dqn_steps/2)),
                  device = gpu
                  )
π_dqn = solve(𝒮_dqn, expert_task)

# Check failure rates after training
failure_rate(expert_task, π_dqn)
mean([failure_rate(t, DQNPolicy(π_dqn.Q, t)) for t in test_tasks])
mean([discounted_return(t, DQNPolicy(π_dqn.Q, t)) for t in test_tasks])

## Fill a buffer with expert trajectories
expert_trajectories = ExperienceBuffer(expert_task, 1000)
fill!(expert_trajectories, expert_task, π_dqn)

sum(expert_trajectories[:r])

# Solve with GAIL
𝒮_gail = GAILSolver(π = DQNPolicy(Qnet(), expert_task), 
                    D = DQNPolicy(Qnet(softmax), expert_task),
                    N = 2000,
                    expert_buffer = expert_trajectories,
                    batch_size = 128,
                    target_update_period = 100,
                    log = LoggerParams(dir = "log/gail", period = 50),
                    exploration_policy = EpsGreedyPolicy(expert_task, LinearDecaySchedule(start=1., stop=0.1, steps=gail_steps/2))
                    )
π_gail = solve(𝒮_gail, expert_task)

# Check failure rates after training
failure_rate(expert_task, π_gail)
mean([failure_rate(t, DQNPolicy(π_gail.Q, t)) for t in test_tasks])
mean([discounted_return(t, DQNPolicy(π_gail.Q, t)) for t in test_tasks])

# Solve with NDA-GAIL
nda_trajectories = gen_buffer(nda_tasks, RandomPolicy(expert_task), nda_buffer_size, desired_return = -1., nonzero_transitions_only = false)
𝒮_nda_gail = GAILSolver(π = DQNPolicy(Qnet(), expert_task), 
                    D = DQNPolicy(Qnet(softmax), expert_task),
                    N = 1000,
                    λ_nda = .5f0,
                    expert_buffer = expert_trajectories,
                    nda_buffer = nda_trajectories,
                    batch_size = 128,
                    exploration_policy = EpsGreedyPolicy(expert_task, LinearDecaySchedule(start=1., stop=0.1, steps=gail_steps/2))
                    )
π_nda_gail = solve(𝒮_nda_gail, expert_task)

failure_rate(expert_task, π_nda_gail)
mean([failure_rate(t, DQNPolicy(π_nda_gail.Q, t)) for t in test_tasks])
mean([discounted_return(t, DQNPolicy(π_nda_gail.Q, t)) for t in test_tasks])

                    
## Make some plots
# Plot on the training MDP
expert_occupancy = gen_occupancy(expert_trajectories, expert_task)
c_expert = render(expert_task, (s = GWPos(7,5),), color = s->reward(expert_task,s) <0 ? -10. :  Float64(expert_occupancy[s]) / 2., policy = π_dqn)
c_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy =π_gail)
c_nda_gail = render(mdp, (s = GWPos(7,5),), color = s->10.0*reward(mdp, s), policy = ChainPolicy(nda_gail_net,mdp))
hstack(c_expert, r, c_gail, r, c_nda_gail) |> SVG("images/mdp1.svg")

c_expert2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(dqn_net, mdp2))
c_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(gail_net, mdp2))
c_nda_gail2 = render(mdp2, (s = GWPos(7,5),), color = s->10.0*reward(mdp2, s), policy = ChainPolicy(nda_gail_net, mdp2))
hstack(c_expert2, r, c_gail2, r, c_nda_gail2) |> SVG("images/mdp2.svg")

nda_occupancy = gen_occupancy(nda_trajectories, mdp2)
c_expert = render(mdp2, (s = GWPos(7,5),), color = s-> Float64(nda_occupancy[s]) / 1.6, policy = ChainPolicy(dqn_net, mdp))


