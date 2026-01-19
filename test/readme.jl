using Crux

mdp = SimpleGridWorld()
S = state_space(mdp)

A() = DiscreteNetwork(Chain(Dense(2, 8, relu), Dense(8, 4)), actions(mdp))

solver = DQN(π=A(), S=S, N=100_000)
policy = solve(solver, mdp)