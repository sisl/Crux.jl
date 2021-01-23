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
    
    export AbstractSpace, DiscreteSpace, ContinuousSpace, useonehot, type, dim, 
           state_space, device, cpucall, gpucall, mdcall, bslice, whiten
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, ExperienceBuffer, minibatch,
           prioritized, dim, episodes, update_priorities!, uniform_sample!, 
           prioritized_sample!
    include("experience_buffer.jl")
    
    include("training.jl")
    
    export NetworkPolicy, polyak_average!, ContinuousNetwork, DiscreteNetwork, 
           ActorCritic, GaussianPolicy, GaussianNoiseExplorationPolicy, FirstExplorePolicy,
           entropy, action_space
    include("policies.jl")
    
    export Sampler, initial_observation, terminate_episode!, step!, steps!, 
           episodes!, fillto!, undiscounted_return, discounted_return, failure, 
           fill_gae!, fill_returns!, trime!
    include("sampler.jl")
    
    export elapsed, LoggerParams, aggregate_info, log_performance, 
           log_discounted_return, log_undiscounted_return, log_failure, 
           log_exploration, smooth, readtb, plot_learning, episode_frames, gif 
    include("logging.jl")
    
    export DiagonalFisherRegularizer, add_fisher_information_diagonal!, update_fisher!
    include("extras/fisher_information.jl")
    
    export MultitaskDecaySchedule, sequential_learning, experience_replay, ewc, log_multitask_performances!
    include("extras/multitask_learning.jl")
    
    export weighted_mean, td_loss, td_error
    include("solvers/value_common.jl")
    
    export DQNSolver
    include("solvers/dqn.jl")
    
    export PGSolver, ppo, reinforce, a2c
    include("solvers/actor_critic.jl")
    
    export GAILSolver
    include("solvers/gail.jl")

    export DDPGSolver
    include("solvers/ddpg.jl")
end

