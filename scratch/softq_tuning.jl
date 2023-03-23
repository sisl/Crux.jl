using POMDPs, Crux, Flux, POMDPGym

## Cartpole - V0
mdp = GymPOMDP(:CartPole, version = :v1)
as = actions(mdp)
S = state_space(mdp)

SoftA(α::Float32) = SoftDiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as;α=α)

# temperature tuning
αs = Vector{Float32}([1,0.5,0.2,0.1])
𝒮_sqls = [SoftQ(π=SoftA(α), S=S, N=10000, interaction_storage=[]) for α in αs]
π_sqls = [@time solve(𝒮_sqls[i], mdp) for i=1:length(αs)]
p = plot_learning(𝒮_sqls, title = "CartPole-V0 SoftQ Tradeoff Curves", 
    labels = ["SQL ΔN=($dn),ep=($e)" for (dn,e) in mix])
Crux.savefig(p, "scratch/cartpole_soft_q_temperature_tradeoffs.pdf")

# collection and c_opt_epoch optimization
ΔNs=[1,2,4]
epochs = [1,5,10,50]
mix = Iterators.product(ΔNs,epochs)  
𝒮_sqls_2 = [SoftQ(π=SoftA(Float32(0.5)), S=S, N=10000, 
    ΔN=dn, c_opt=(;epochs=e), interaction_storage=[]) for (dn,e) in mix]
π_sqls_2 = [@time solve(x, mdp) for x in 𝒮_sqls_2]
p = plot_learning(𝒮_sqls_2, title = "CartPole-V0 SoftQ Tradeoff Curves", 
    labels = ["SQL ΔN=($dn),ep=($e)" for (dn,e) in mix])
Crux.savefig(p, "scratch/cartpole_soft_q_sampling_tradeoffs.pdf")


