using Revise
using POMDPs, Crux, Flux, POMDPGym

## Pendulum
mdp = PendulumMDP(actions=[-2., -0.5, 0, 0.5, 2.]) # Continuous
# mdp = ContinuousBanditMDP(2.0) # Continuous (TODO: own file `continuous_bandit.jl`)
S = state_space(mdp)

# Define the networks we will use
Qₚₚₒ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1), x -> 200f0.*x .- 200f0)

# Randomly initialize critic network 𝑄(s, a | θᶜ) and actor μ(s | θᵘ) with weights θᶜ and θᵘ
Q() = Chain(x -> x ./ [6.3f0, 8f0, 2f0], Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 1), x -> 200f0.*x .- 200f0) # NOTE change to 3: vcat(s,a)
μ() = Chain(x -> x ./ [6.3f0, 8f0], Dense(2, 64, relu), Dense(64, 64, relu), Dense(64, 1))
# Q() = Chain(Dense(2, 64, relu), Dense(64, 1))
# Q() = Chain(x->.-(x .- 2).^2)
# μ() = Chain(Dense(1, 64, relu), Dense(64, 1))

# Initialize target network 𝑄′ and μ′ with weights θᶜ′ ⟵ θᶜ, and θᵘ ⟵ θᵘ
Q′() = Q()
μ′() = μ()

# @info "Solving with PPO"
# 𝒮_ppo = PGSolver(π=ActorCritic(GaussianPolicy(μ=μ(), logΣ=zeros(Float32, 1)), Qₚₚₒ()),
#                  S=S, N=100000, ΔN=2048, loss=ppo(), opt=Flux.Optimiser(ClipNorm(1f0), ADAM(1e-3)),
#                  batch_size=512, epochs=100)
# π_ppo = solve(𝒮_ppo, mdp)

@info "Solving with DDPG"
𝒮_ddpg = DDPGSolver(π=ActorCritic(μ(), Q()) |> gpu,
                    π′=ActorCritic(μ′(), Q′()) |> gpu,
                    S=S, N=100_000)
π_prev = deepcopy(𝒮_ddpg.π.A)
# 𝒮_ddpg = DDPGSolver(π=ActorCritic(GaussianPolicy(μ=μ(), logΣ=zeros(Float32, 1)), Q()),
#                     π′=ActorCritic(GaussianPolicy(μ=μ′(), logΣ=zeros(Float32, 1)), Q′()),
#                     S=S, N=100_000, batch_size=512)
π_ddpg = solve(𝒮_ddpg, mdp)

# Plot the learning curve
# p = plot_learning([𝒮_ppo, 𝒮_ddpg], title="Pendulum Swingup Training Curves", labels=["PPO", "DDPG"])
p = plot_learning([𝒮_ddpg], title="Pendulum Swingup Training Curves", labels=["DDPG"])

# Produce a gif with the final policy
# gif(mdp, π_ppo, "pendulum_ppo.gif", max_steps=200)
# gif(mdp, π_ddpg, "pendulum_ddpg.gif", max_steps=200)

# Return plot
p
