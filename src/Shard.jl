module Shard
    using Random
    using Distributions
    using POMDPs
    using POMDPPolicies
    using POMDPSimulators
    using Parameters
    using TensorBoardLogger
    using Flux
    using Flux.Optimise: train!
    using CUDA
    using LinearAlgebra
    using ValueHistories
    using DataStructures

    export sdim, adim, device, todevice
    include("utils.jl")
    
    export inverse_query, mdp_data, circular_indices, ExperienceBuffer, capacity, 
           prioritized, update_priorities!, uniform_sample!, prioritized_sample!
    include("experience_buffer.jl")
    
    export sync!, Baseline, network, logits, 
           DQNPolicy, CategoricalPolicy, GaussianPolicy
    include("policies.jl")
    
    export Sampler, explore, step!, steps!, episodes!, fillto!m, 
           undiscounted_return, discounted_return, failure, fill_gae!, fill_returns!
    include("sampler.jl")
    
    export LoggerParams, log_perforamnce, log_discounted_return, 
           log_undiscounted_return, log_failure, log_val, log_loss, log_gradient, 
           log_exploration, smooth, readtb, plot_learning_curves
    include("logging.jl")
    
    export MultitaskDecaySchedule, sequential_learning, experience_replay
    include("multitask_learning.jl")
    
    export weighted_mean, target, q_predicted, td_loss, td_error
    include("value_common.jl")
    
    export DQNSolver
    include("dqn.jl")
    
    export VPGSolver
    include("vpg.jl")
    
    export DQNGAILSolver
    include("dqn_gail.jl")
end

