module Crux
    using Random
    import Distributions:Categorical, MvNormal, logpdf
    using POMDPs
    using POMDPPolicies
    using POMDPModelTools
    using Parameters
    using TensorBoardLogger
    using Flux
    using Zygote
    using Flux.Optimise: train!
    using CUDA
    using LinearAlgebra
    using ValueHistories
    using DataStructures
    using Plots
    using ColorSchemes
    import Images: save
    using Base.Iterators: partition
    
    export type, dim, useonehot, state_space, ContinuousSpace, DiscreteSpace, AbstractSpace, device, mdcall, bslice, cpucall, gpucall
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, ExperienceBuffer, episodes, shuffle!,
           minibatch, prioritized, update_priorities!, uniform_sample!, prioritized_sample!
    include("experience_buffer.jl")
    
    export sync!, control_features, network, logits, action_space, ActorCritic,
           DQNPolicy, CategoricalPolicy, GaussianPolicy, logpdf
    include("policies.jl")
    
    export Sampler, explore, terminate_episode!, step!, steps!, episodes!, fillto!, 
           undiscounted_return, discounted_return, failure, fill_gae!, fill_returns!
    include("sampler.jl")
    
    export elapsed, LoggerParams, log_perforamnce, log_discounted_return, 
           log_undiscounted_return, log_failure, log_val, log_loss, log_gradient, 
           log_exploration, smooth, readtb, plot_learning, episode_frames,
           gif
    include("logging.jl")
    
    export init_fisher_diagonal, add_fisher_information_diagonal!, update_fisher_diagonal!
    include("extras/fisher_information.jl")
    
    export MultitaskDecaySchedule, sequential_learning, experience_replay, ewc
    include("extras/multitask_learning.jl")
    
    export weighted_mean, target, q_predicted, td_loss, td_error
    include("solvers/value_common.jl")
    
    export DQNSolver
    include("solvers/dqn.jl")
    
    export DQNGAILSolver
    include("solvers/dqn_gail.jl")
    
    export PGSolver, ppo, reinforce, a2c
    include("solvers/actor_critic.jl")
    
end

