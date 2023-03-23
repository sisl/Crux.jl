using POMDPs, Crux, Flux, POMDPGym
using Random
using BSON
using Debugger

## Problems
problems = Dict(
    "cartpole"=>GymPOMDP(:CartPole, version = :v0),
    "pendulum"=> PendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.]),
    "inverted_pendulum"=>InvertedPendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.]),
    "daa"=> DetectAndAvoidMDP(),
    "lunar_lander"=>GymPOMDP(:LunarLander, version = :v2),
)

expert_sizes = [8, 16, 32, 64, 128, 1024]

h = 32
QS(sdim,as,h) = DiscreteNetwork(Chain(Dense(sdim, h, relu), Dense(h, h, relu), Dense(h, length(as))), as)

for name in keys(problems)

    println("$(name) imitation")

    mdp = problems[name]
    as = [actions(mdp)...]
    S = state_space(mdp) #, σ=[3.14f0, 8f0])
    sdim = sum(Crux.dim(S))
    γ = Float32(discount(mdp))
    expert_trajectories = BSON.load("examples/il/expert_data/$(name)_discrete.bson")[:data]


    for es in expert_sizes
        bs=min(128, 2*es)

        # subsample expert demonstrations
        demos = buffer_like(expert_trajectories, capacity=es, device=device(expert_trajectories))
        rand!(demos, expert_trajectories)

        # Solve with Behavioral Cloning
        𝒮_bc = BC(π=QS(sdim,as,h), 𝒟_demo=demos, S=S, opt=(epochs=Int(10000*es/bs),batch_size=bs), 
            window=1000,log=(period=10,))
        solve(𝒮_bc, mdp)

        # Solve with IQ-Learn
        𝒮_iql = OnlineIQLearn(π=QS(sdim,as,h), 𝒟_demo=demos, S=S, γ=γ, N=50000, ΔN=1, 
            c_opt=(;epochs=1,batch_size=bs),reg=false,gp=false, log=(;period=50))
        solve(𝒮_iql, mdp)

        # Plot true rewards
        p = plot_learning([𝒮_bc, 𝒮_iql], title = "$(name) Imitation Comparison: $(es) demos ", 
            labels = ["BC", "IQLearn"])
        Crux.savefig(p, "scratch/figs/$(name)_$(es)_imitation_training.pdf")


    end
end