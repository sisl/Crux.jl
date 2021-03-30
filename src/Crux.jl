module Crux
    using Random
    using Distributions
    using POMDPs
    using POMDPModelTools:render
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
    
    export AbstractSpace, DiscreteSpace, ContinuousSpace, type, dim, 
           state_space, device, cpucall, gpucall, mdcall, bslice, whiten, to2D
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, PriorityParams, ExperienceBuffer, buffer_like, minibatch,
           clear!, isprioritized, dim, episodes, update_priorities!, uniform_sample!, 
           prioritized_sample!, capacity, geometric_sample!
    include("experience_buffer.jl")
    
    export TrainingParams, batch_train!
    include("training.jl")
    
    export NetworkPolicy, polyak_average!, ContinuousNetwork, DiscreteNetwork, 
           DoubleNetwork, ActorCritic, GaussianPolicy, SquashedGaussianPolicy,
           GaussianNoiseExplorationPolicy, FirstExplorePolicy, ϵGreedyPolicy, LinearDecaySchedule,
           entropy, logpdf, action_space, exploration, layers
    include("policies.jl")
    
    export Sampler, initial_observation, terminate_episode!, step!, steps!, 
           episodes!, fillto!, undiscounted_return, discounted_return, failure, 
           fill_gae!, fill_returns!, trime!
    include("sampler.jl")
    
    export elapsed, LoggerParams, aggregate_info, log_performance, 
           log_discounted_return, log_undiscounted_return, log_failure, 
           log_exploration
    include("logging.jl")
    
    export smooth, readtb, plot_learning, episode_frames, gif, percentile, 
           find_crossing, plot_jumpstart, directories, plot_peak_performance, 
           plot_learning, plot_cumulative_rewards, plot_steps_to_threshold
    include("analysis.jl")
    
    export DenseSN, ConvSN
    include("extras/spectral_normalization.jl")
    
    export GAN_BCELoss, GAN_LSLoss, GAN_HingeLoss, GAN_WLossGP, GAN_WLoss, Lᴰ, Lᴳ
    include("extras/gans.jl")
    
    export gradient_penalty
    include("extras/gradient_penalty.jl")
    
    export DiagonalFisherRegularizer, add_fisher_information_diagonal!, update_fisher!
    include("extras/fisher_information.jl")
    
    export BatchRegularizer, value_regularization, action_regularization, action_value_regularization
    include("extras/batch_regularization.jl")
    
    export OrthogonalRegularizer
    include("extras/orthogonal_regularization.jl")
    
    export MultitaskDecaySchedule, log_multitask_performances!, continual_learning
    include("extras/multitask_learning.jl")
    
    export OnPolicySolver, OffPolicySolver
    export REINFORCE, A2C, PPO, DQN, DDPG, TD3, SAC
    export GAIL, ValueDICE, BC, AdVIL
    export mse_action_loss, logpdf_bc_loss, mse_value_loss
    include("model_free/on_policy.jl")
    include("model_free/off_policy.jl")
    include("model_free/rl/reinforce.jl")
    include("model_free/rl/a2c.jl")
    include("model_free/rl/ppo.jl")
    include("model_free/rl/dqn.jl")
    include("model_free/rl/ddpg.jl")
    include("model_free/rl/td3.jl")
    include("model_free/rl/sac.jl")
    include("model_free/il/bc.jl")
    include("model_free/il/AdVIL.jl")
    include("model_free/il/gail.jl")
    include("model_free/il/valueDICE.jl")
end

