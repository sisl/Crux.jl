function test_solver(S, \pi, mdp)
    # run it once
    
    # run it again with the same rng
    
    # run it again on the gpu with the same rng
    
    # compare the results
    
end

discrete_mdp = GymPOMDP(:CartPole, version = :v0)
continuous_mdp = PendulumMDP()

# RL solvers
test_solver(DQN, )
test_solver(PPO)
test_solver(REINFORCE)
test_solver(A2C)
test_solver(DDPG)
test_solver(TD3)
test_solver(SAC)


# IL Solvers
test_solver(GAIL)
test_solver(valueDICE)

