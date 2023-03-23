using Crux, Flux, POMDPGym, Random, POMDPs, BSON

## Cartpole
mdp = GymPOMDP(:CartPole, version = :v0)
as = actions(mdp)
S = state_space(mdp)
γ = Float32(discount(mdp))

SA() = SoftDiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as;α=Float32(1.))

# Fill a buffer with expert trajectories
expert_trajectories = BSON.load("examples/il/expert_data/cartpole.bson")[:data]

# IQLearn ΔN, c_opt epochs hyperparams
mix = [(1,1),(1,5), (4,5), (4,10), (20,20), (20,50)]
𝒮_iqls = [OnlineIQLearn(π=SA(), 𝒟_demo=expert_trajectories, S=S, γ=γ, N=10000, ΔN=dn, log=(;period=100), c_opt=(;epochs=e)) for (dn,e) in mix]
[@time solve(i, mdp) for i in 𝒮_iqls]
p = plot_learning(𝒮_iqls, title = "CartPole-V0 IQL Tradeoff Curves", 
    labels = ["IQL ΔN=($dn),ep=($e)" for (dn,e) in mix])
Crux.savefig(p, "scratch/cartpole_iqlearn_dne_tradeoffs.pdf")

# IQLearn λ_gp, α_reg hyperparams
λ_gps = Float32[1, 0.1, 0.01, 0.]
α_regs = Float32[100, 1, 0.1]
mix = Iterators.product(λ_gps,α_regs)
𝒮_iqls = [OnlineIQLearn(π=SA(), 𝒟_demo=expert_trajectories, 
    S=S, γ=γ, λ_gp=i, α_reg=j, N=10000, ΔN=1, 
    log=(;period=100), c_opt=(;epochs=1)) for (i,j) in mix]
[@time solve(i, mdp) for i in 𝒮_iqls]
p = plot_learning(𝒮_iqls, title = "CartPole-V0 IQL Tradeoff Curves", 
    labels = ["IQL λ_gp=($i),α_reg=($j)" for (i,j) in mix])
Crux.savefig(p, "scratch/cartpole_iqlearn_reg_tradeoffs.pdf")