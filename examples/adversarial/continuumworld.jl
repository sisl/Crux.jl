using POMDPs, Crux, Flux, POMDPGym, Random, Distributions, POMDPPolicies
Crux.set_function("isfailure", POMDPGym.isfailure)

struct MvNormal2
    μ
    logσ
    μ_gpu
    logσ_gpu
    d
    MvNormal2(μ, σ) = new(μ, Base.log.(σ), μ|>gpu, Base.log.(σ)|>gpu, MvNormal(cpu(μ), cpu(σ)))
end

function Base.rand(d::MvNormal2, sz::Int...) 
    r = rand(d.d, sz...)  
    if length(r) == 1
        return r[1]
    end
    r
end

function Distributions.logpdf(d::MvNormal2, x)
    if device(x)==gpu
        Crux.gaussian_logpdf(d.μ_gpu, d.logσ_gpu, x)
    else
        Crux.gaussian_logpdf(d.μ, d.logσ, x)
    end
end
        

# Define the disturbance distribution based on a normal distribution
px = MvNormal2([0f0, 0f0], [0.2f0, 0.2f0])

# Construct the MDP
function build_mdp()
    Random.seed!(1)
    rp() = rand(Uniform(-1, 1))
    rad() = abs(rand(Uniform(0.05, 0.15)))

    failures = Dict(Circle(Vec2(rp(),rp()), rad())=>-0.01f0 for i=1:1)
    rewards = Dict(failures..., Circle(Vec2(-0.5f0,0.5f0), 0.2f0)=>0.9f0)
    mdp = AdditiveAdversarialMDP(ContinuumWorldMDP(rewards=rewards, vel_thresh=1f0, discount=1.0), px)
end
mdp = build_mdp()
render(mdp)
S = state_space(mdp)

# construct the models for the protagonist
QSA() = ContinuousNetwork(Chain(Dense(S.dims[1] + 2, 128, relu), Dense(128, 128, relu), Dense(128, 1)))
A() = ContinuousNetwork(Chain(Dense(S.dims[1], 128, relu), Dense(128, 128, relu), Dense(128, 2, tanh)))
Protag() = ActorCritic(A(), DoubleNetwork(QSA(), QSA()))

# And the models for the antagonist
Pf() = ContinuousNetwork(Chain(Dense(S.dims[1] + 2, 128, relu), Dense(128, 128, relu), Dense(128, 1, sigmoid)))
function G()
    base = Chain(Dense(S.dims[1], 128, relu), Dense(128, 128, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(128, 2)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(128, 2)))
    GaussianPolicy2(mu, logΣ, true)
end
Antag() = ActorCritic(G(), Pf())

# Solve with a traditional RL approach
TD3_params() = (N=100000, S=S, buffer_size=Int(1e5), buffer_init=1000)

𝒮_td3 = TD3(;π=ActorCritic(A(), DoubleNetwork(QSA(), QSA())), TD3_params()...)
π_td3 = solve(𝒮_td3, mdp)

# Solver with adversarial approach
ARL_params() = (TD3_params()..., desired_AP_ratio=1.0)

# solve with IS
𝒮_isarl = ISARL_Continuous(;π=AdversarialPolicy(π_td3, Antag()) |> gpu, ARL_params()..., px)
π_isarl = solve(𝒮_isarl, mdp)

# solve with RARL
𝒮_rarl = Crux.RARL_Continuous(;π=AdversarialPolicy(Protag(), Protag(), GaussianNoiseExplorationPolicy(0.2f0)) |>gpu, ARL_params()...)
π_rarl = solve(𝒮_rarl, mdp)

Dvanilla = steps!(Sampler(mdp, π_td3, S=S, required_columns=[:fail]), Nsteps=10000)
sum(Dvanilla[:fail])

Disarl = steps!(Sampler(mdp, protagonist(π_isarl), S=S, required_columns=[:fail]), Nsteps=10000)
sum(Disarl[:fail])

Drarl = steps!(Sampler(mdp, protagonist(π_rarl), S=S, required_columns=[:fail]), Nsteps=10000)
sum(Drarl[:fail])

gif(mdp, protagonist(π_isarl), "isarl.gif", Neps=10)
gif(mdp, π_td3, "td3.gif", Neps=100)

