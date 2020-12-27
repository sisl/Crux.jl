using Crux, Flux, POMDPGym
using Test

# Setup the problem parameters
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
N_nda = 10
N_test = 10
expert_task = SimpleGridWorld(size = sz, tprob = 1.0, rewards = random_lava(sz, 1, goal = (7,5), rng = MersenneTwister(0)))
nda_tasks = [SimpleGridWorld(size = sz, tprob = 1.0,  rewards = random_lava(sz, 1, goal = (7,5))) for _=1:N_nda]
test_tasks = [SimpleGridWorld(size = sz, tprob = 1.0,  rewards = random_lava(sz, 1, goal = (7,5))) for _=1:N_test]

simple_display(expert_task)
simple_display(nda_tasks[3])
simple_display(test_tasks[9])

Qnet() = Chain(Dense(input_dim, 128, relu), Dense(128,64, relu), Dense(64, 4))
as = actions(expert_task) 

dqn_steps = 20000 # to learn an expert policy
gail_steps = 2000
expert_buffer_size = 1000 
nda_buffer_size = 1000
λ_nda = 0.5f0 # Constant for NDA. λ = 1 ignores the NDA trajectories

## solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Qnet(), as), sdim = input_dim, N=dqn_steps, batch_size = 128)
π_dqn = solve(𝒮_dqn, expert_task)

# Check failure rates after training
failure(expert_task, π_dqn)
mean([failure(t, π_dqn) for t in test_tasks])
mean([failure(t, π_dqn) for t in test_tasks])

## Fill a buffer with expert trajectories
expert_trajectories = ExperienceBuffer(steps!(Sampler(mdp = expert_task, π = π_dqn), Nsteps = expert_buffer_size))
sum(expert_trajectories[:r])

# Solve with GAIL
𝒮_gail = DQNGAILSolver(π = DQNPolicy(Qnet(), as), 
                       D = DQNPolicy(Qnet(), as), 
                       sdim = input_dim,
                       N = gail_steps,
                       expert_buffer = expert_trajectories,
                       batch_size = 128,
                       Δtarget_update = 100,
                       Δtrain = 1,
                       )
π_gail = solve(𝒮_gail, expert_task)

# Check failure rates after training
failure(expert_task, π_gail)
mean([failure(t, π_gail) for t in test_tasks])
mean([discounted_return(t, π_gail) for t in test_tasks])

