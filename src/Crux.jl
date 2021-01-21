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
    import Zygote: ignore
    using Flux.Optimise: train!
    using CUDA
    using LinearAlgebra
    using ValueHistories
    using DataStructures
    using Plots
    using ColorSchemes
    import Images: save
    using Statistics
    using Base.Iterators: partition
    
    export type, dim, useonehot, state_space, ContinuousSpace, DiscreteSpace,  
           AbstractSpace, device, mdcall, bslice, cpucall, gpucall, whiten, polyak_average!
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, ExperienceBuffer, episodes, shuffle!,
           minibatch, prioritized, update_priorities!, uniform_sample!, prioritized_sample!
    include("experience_buffer.jl")
    
    include("training.jl")
    
    export logits, action_space, logpdf, entropy, update_target!, target_value, target_action,
           ActorCritic, DQNPolicy, CategoricalPolicy, GaussianPolicy, GaussianNoiseExplorationPolicy,  DeterministicNetwork, DDPGPolicy
    include("policies.jl")
    
    export Sampler, explore, terminate_episode!, step!, steps!, episodes!, fillto!, 
           undiscounted_return, discounted_return, failure, fill_gae!, fill_returns!
    include("sampler.jl")
    
    export elapsed, LoggerParams, log_performance, log_discounted_return, 
           log_undiscounted_return, log_failure, log_val, log_loss, log_gradient, 
           log_exploration, smooth, readtb, plot_learning, episode_frames,
           gif, aggregate_info
    include("logging.jl")
    
    export DiagonalFisherRegularizer, add_fisher_information_diagonal!, update_fisher!
    include("extras/fisher_information.jl")
    
    export MultitaskDecaySchedule, sequential_learning, experience_replay, ewc
    include("extras/multitask_learning.jl")
    
    export weighted_mean, td_loss, td_error
    include("solvers/value_common.jl")
    
    export DQNSolver, DQN_target
    include("solvers/dqn.jl")
    
    export PGSolver, ppo, reinforce, a2c
    include("solvers/actor_critic.jl")
    
    export GAILSolver
    include("solvers/gail.jl")

    export DDPGSolver, DDPG_target
    include("solvers/ddpg.jl")
end

