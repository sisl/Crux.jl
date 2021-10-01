using POMDPs # This defines the interface for mdps
using POMDPPolicies # Used just for the "FunctionPolicy" type for rendering
using POMDPGym # This is where the environments are defined (including rendering code)
using Crux # This the deep RL library with the various solvers
using Flux # This is the ML library for defining models

# Define the lavaworld mdp
mdp = LavaWorldMDP(lava = [GWPos(2,5), GWPos(3,5), GWPos(4,5), GWPos(5,5)], goal=GWPos(7,5))
S = state_space(mdp) # this is the state space of the mdp
as = actions(mdp) # Actions of the mdp
idim = prod(Crux.dim(S)) # input dimension

# See the lavaworld rendered
render(mdp)

# Define the Q network and solve with DQN
Q() = DiscreteNetwork(Chain(flatten, Dense(idim, 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
solver = DQN(π=Q(), S=S, N=10000)
π_exp = solve(solver, mdp)

# Record some expert trajectories
Neps = 10
B_expert = ExperienceBuffer(episodes!(Sampler(mdp, π_exp), Neps=Neps))
mean_return = sum(B_expert[:r]) / Neps


## Lets see how our expert does
# Plot the policy
render(mdp, policy=π_exp, return_compose=true)

# Plot the value function (The factor of 10 is a quirk of an external render function that assumes values are between -10 and 10)
render(mdp, color = s->10.0*maximum(value(π_exp, convert_s(Vector{Float32}, s, mdp))))

# Plot the occupancy of different grid points in the expert data
occupancy = POMDPGym.gen_occupancy(B_expert, mdp)
render(mdp, color = s->200f0*occupancy[s])

## Now lets solve using imitation learning

# Start with a simple BC baseline
𝒮_bc = BC(π=Q(), S=S, 𝒟_demo=B_expert, opt=(;epochs=10000), log=(;period=100,))
π_bc = solve(𝒮_bc, mdp)
render(mdp, policy=π_bc, return_compose=true)

# then use off-policy GAIL
D() = ContinuousNetwork(Chain(flatten, DenseSN(idim + 4, 64, relu), DenseSN(64, 64, relu), DenseSN(64, 2)))
𝒮_gail = OffPolicyGAIL(D=D(), 
                       𝒟_demo=B_expert, 
                       solver=DQN, 
                       π=Q(), 
                       S=S,
                       N=10000,
                       )
π_gail = solve(𝒮_gail, mdp)

render(mdp, policy=π_gail, return_compose=true)

# Now lets see how it does with a different lava configuration:
mdp2 = LavaWorldMDP(lava = [GWPos(2,3), GWPos(3,3), GWPos(4,3), GWPos(5,3)], goal=GWPos(7,5))
render(mdp2, policy=π_gail, return_compose=true)

