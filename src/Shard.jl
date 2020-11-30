module Shard
    using Random
    using Distributions
    using POMDPs
    using POMDPPolicies
    using Parameters
    using TensorBoardLogger
    using Flux
    using Flux.Optimise: train!
    using CUDA
    using LinearAlgebra
    using ValueHistories
    using DataStructures
    using Plots
    using ColorSchemes
    using Base.Iterators: partition
    
    export sdim, adim, device, todevice
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, circular_indices, ExperienceBuffer, 
           minibatch, prioritized, update_priorities!, uniform_sample!, prioritized_sample!
    include("experience_buffer.jl")
    
    export sync!, Baseline, network, logits, 
           DQNPolicy, CategoricalPolicy, GaussianPolicy
    include("policies.jl")
    
    export Sampler, explore, terminate_episode!, step!, steps!, episodes!, fillto!, 
           undiscounted_return, discounted_return, failure, fill_gae!, fill_returns!
    include("sampler.jl")
    
    export elapsed, LoggerParams, log_perforamnce, log_discounted_return, 
           log_undiscounted_return, log_failure, log_val, log_loss, log_gradient, 
           log_exploration, smooth, readtb, plot_learning_curves
    include("logging.jl")
    
    export MultitaskDecaySchedule, sequential_learning, experience_replay
    include("solvers/multitask_learning.jl")
    
    export weighted_mean, target, q_predicted, td_loss, td_error
    include("solvers/value_common.jl")
    
    export DQNSolver
    include("solvers/dqn.jl")
    
    export VPGSolver
    include("solvers/vpg.jl")
    
    export DQNGAILSolver
    include("solvers/dqn_gail.jl")
end

