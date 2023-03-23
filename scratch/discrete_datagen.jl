using POMDPs, Crux, Flux, POMDPGym
using Random
using BSON

## Problems
problems = Dict(
    "cartpole"=>GymPOMDP(:CartPole, version = :v0),
    "pendulum"=> PendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.]),
    "inverted_pendulum"=>InvertedPendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.]),
    "daa"=> DetectAndAvoidMDP(),
    "acrobot"=>GymPOMDP(:Acrobot, version = :v1), # doesnt work
    "mountain_car"=>GymPOMDP(:MountainCar, version = :v0), #doesnt work
    "lunar_lander"=>GymPOMDP(:LunarLander, version = :v2),
)

# for lunar lander to work, had to add box2d-py:
# using Conda; Conda.add("box2d-py";channel="conda-forge")

# Define the networks we will use
h = 32
QS(sdim,as,h) = DiscreteNetwork(Chain(Dense(sdim, h, relu), Dense(h, h, relu), Dense(h, length(as))), as)

train_steps = 30000

for name in keys(problems)

    mdp = problems[name]
    as = [actions(mdp)...]
    S = state_space(mdp) #, σ=[3.14f0, 8f0])
    sdim = sum(Crux.dim(S))

    # Solve with DQN (gets to > -200 reward, ~30 sec)
    𝒮_dqn = DQN(π=QS(sdim,as,h), S=S, ΔN=1, c_opt=(;epochs=5), N=train_steps)
    @time π_dqn = solve(𝒮_dqn, mdp)

    # Solve with SoftQ
    𝒮_sql = SoftQ(π=QS(sdim,as,h), α=Float32(0.2), ΔN=1, c_opt=(;epochs=5), S=S, N=train_steps)
    @time π_sql = solve(𝒮_sql, mdp)

    # Plot the learning curve
    p = plot_learning([𝒮_dqn, 𝒮_sql, ], title="$(name) Training Curves", 
        labels=["DQN", "SQL"], legend=:right)
    Crux.savefig("scratch/figs/$(name)_discrete_benchmark.pdf")

    ## Save data for imitation learning
    s = Sampler(mdp, 𝒮_dqn.agent, max_steps=200, required_columns=[:t])

    data = steps!(s, Nsteps=10000)
    data[:expert_val] = ones(Float32, 1, 10000)

    data = ExperienceBuffer(data)
    BSON.@save "examples/il/expert_data/$(name)_discrete.bson" data

end
