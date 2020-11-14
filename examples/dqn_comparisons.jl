include("../src/includes.jl") # this will be replaced with a using statement eventually
using DeepQLearning
using POMDPModels, POMDPModelTools
POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp = rand(transition(mdp, s, a )), r = reward(mdp, s, a))
function POMDPs.initialstate(mdp::SimpleGridWorld)
    while true
        x, y = rand(1:mdp.size[1]), rand(1:mdp.size[2])
        !(GWPos(x,y) in mdp.terminate_from) && return Deterministic(GWPos(x,y))
    end
end 

POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, mdp::SimpleGridWorld) = Float32.([s...])


mdp = SimpleGridWorld(size = (10,10), tprob = .7)
s = convert_s(AbstractArray, rand(initialstate(mdp)), mdp)

N = 100000
Q() = Chain(x -> (x .- 5.f0 ) ./ 5.f0, Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 4))

## Maxime's implementation
dqn_solver = DeepQLearningSolver(qnetwork = Q(),
                                 exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=N/2)),
                                 max_steps = N,
                                 dueling = false, 
                                 double_q = false, 
                                 prioritized_replay = false, 
                                 target_update_freq = 2000,
                                 train_freq = 1,
                                 batch_size = 32,
                                 learning_rate = 1f-3,
                                 logdir="log/dqn")
@time pol = solve(dqn_solver, mdp)

## My implementation
gpusolver = DQNSolver(Q = Q(), N=N, batch_size = 32, exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=N/2)), device = gpu)
@time gpu_pol = solve(gpusolver, mdp)

cpusolver = DQNSolver(Q = Q(), N=N, batch_size = 32, exploration_policy = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1., stop=0.1, steps=N/2)))
solve(cpusolver, mdp)



