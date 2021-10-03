using POMDPs, Crux, Flux, POMDPGym, Random, Distributions, POMDPPolicies, PyCall
init_mujoco_render() # Required for visualization
pyimport("safety_gym")
Crux.set_function("isfailure", POMDPGym.isfailure)

# Define the disturbance distribution based on a normal distribution
px = MvNormal([0f0, 0f0], [0.2f0, 0.2f0])

# Construct the MDP
mdp = AdditiveAdversarialPOMDP(EpisodicSafetyGym(pomdp=GymPOMDP(Symbol("Safexp-PointGoal1"), version=:v0), λ=0.01), px)
mdp_nom = AdditiveAdversarialPOMDP(EpisodicSafetyGym(pomdp=GymPOMDP(Symbol("Safexp-PointGoal1"), version=:v0), λ=0.01), px)
# mdp = AdditiveAdversarialPOMDP(GymPOMDP(Symbol("Safexp-PointGoal1"), version=:v0), px)
S = state_space(mdp)

# construct the models for the protagonist
QSA() = ContinuousNetwork(Chain(Dense(62, 256, relu), Dense(256, 256, relu), Dense(256, 1)))
A() = ContinuousNetwork(Chain(Dense(60, 256, relu), Dense(256, 256, relu), Dense(256, 2, tanh)))
Protag() = ActorCritic(A(), DoubleNetwork(QSA(), QSA()))

# And the models for the antagonist
Pf() = ContinuousNetwork(Chain(Dense(62, 256, relu), Dense(256, 256, relu), Dense(256, 1, (x)-> -softplus(-(x-2)))))
function SquashedG()
    base = Chain(Dense(60, 256, relu), Dense(256, 256, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(256, 2)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(256, 2)))
    SquashedGaussianPolicy(mu, logΣ)
end
Antag() = ActorCritic(SquashedG(), Pf())

# Solve with a traditional RL approach
TD3_params() = (N=100000, S=S, buffer_size=Int(1e5), buffer_init=1000, max_steps=1000)

𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), TD3_params()...)
π_td3 = solve(𝒮_td3, mdp)

# Solver with adversarial approach
ARL_params() = (TD3_params()..., desired_AP_ratio=1.0)

# solve with IS
𝒮_isarl = ISARL_Continuous(;π=AdversarialPolicy(Protag(), Antag()), ARL_params()..., px)
π_isarl = solve(𝒮_isarl, mdp, mdp_nom)

# solve with RARL
𝒮_rarl = Crux.RARL_Continuous(;π=AdversarialPolicy(Protag(), Protag(), GaussianNoiseExplorationPolicy(0.2f0)), ARL_params()...)
π_rarl = solve(𝒮_rarl, mdp, mdp_nom)



D_vanilla = steps!(Sampler(mdp, π_td3, S=S, required_columns=[:cost], max_steps=1000), Nsteps=5000)
sum(D_vanilla[:r])/5
sum(D_vanilla[:cost])


D_isarl = steps!(Sampler(mdp, protagonist(π_isarl), S=S, required_columns=[:cost], max_steps=1000), Nsteps=5000)
sum(D_isarl[:r])/5
sum(D_isarl[:cost])

s = tovec(𝒮_isarl.buffer[:s][:, 100], S)
antagonist(π_isarl).A.μ(s)
exp.(antagonist(π_isarl).A.logΣ(s))

using Plots
histogram(𝒮_isarl.buffer[:x][:])




