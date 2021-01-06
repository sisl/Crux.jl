using Crux, Flux, POMDPGym, Random, POMDPs
using Test

# Setup the problem parameters
sz = (7,5)
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
N_nda = 10
N_test = 10
expert_task = LavaWorldMDP(rng = MersenneTwister(0))
nda_tasks = [LavaWorldMDP() for _=1:N_nda]
test_tasks = [LavaWorldMDP() for _=1:N_test]

S = state_space(expert_task)
# render(expert_task)
# render(nda_tasks[3])
# render(test_tasks[9])
Q() = Chain(x -> reshape(x, 105, :), Dense(105, 128, relu), Dense(128,64, relu), Dense(64, 4))
as = [actions(expert_task)...]

dqn_steps = 1000 # to learn an expert policy
gail_steps = 100
expert_buffer_size = 1000 
nda_buffer_size = 1000
λ_nda = 0.5f0 # Constant for NDA. λ = 1 ignores the NDA trajectories

## solve with DQN
𝒮_dqn = DQNSolver(π = DQNPolicy(Q=Q(), actions = as), S=S, N=dqn_steps, batch_size = 128)
π_dqn = solve(𝒮_dqn, expert_task)

## Fill a buffer with expert trajectories
expert_trajectories = ExperienceBuffer(steps!(Sampler(mdp = expert_task, S = S, A = action_space(π_dqn), π = π_dqn), Nsteps = expert_buffer_size))
sum(expert_trajectories[:r])

# Solve with GAIL
𝒮_gail = DQNGAILSolver(π = DQNPolicy(Q = Q(), actions = as), 
                       D = DQNPolicy(Q = Q(), actions = as), 
                       S = S,
                       N = gail_steps,
                       expert_buffer = expert_trajectories,
                       batch_size = 128,
                       Δtarget_update = 100,
                       Δtrain = 1,
                       )
π_gail = solve(𝒮_gail, expert_task)
