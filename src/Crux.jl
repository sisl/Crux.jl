module Crux
    CRUX_WARNINGS = true
    
    set_crux_warnings(val::Bool) = global CRUX_WARNINGS = val
    
    using Random
    using Distributions
    using POMDPs
    using POMDPModelTools:render
    using Parameters
    using TensorBoardLogger
    using Flux
    using Zygote
    import Zygote: ignore_derivatives
    using Flux.Optimise: train!
    using CUDA
    using LinearAlgebra
    using ValueHistories
    using Plots
    using ColorSchemes
    import Images: save
    using Statistics
    using StatsBase
    using Base.Iterators: partition
    using WeightsAndBiasLogger
    using Dates
    
    extra_functions = Dict()
    function set_function(key, val)
        extra_functions[key] = val
    end
    
    export device, cpucall, gpucall, mdcall, bslice
    include("devices.jl")
    
    export AbstractSpace, DiscreteSpace, ContinuousSpace, type, dim, state_space
    include("spaces.jl")
    
    export ObjectCategorical, ConstantLayer, whiten, to2D, tovec, 
           LinearDecaySchedule, MultitaskDecaySchedule, multi_td_loss, multi_actor_loss
    include("utils.jl")
    
    export MinHeap, inverse_query, mdp_data, PriorityParams, ExperienceBuffer, buffer_like, minibatch,
           clear!, trim!, isprioritized, dim, episodes, update_priorities!, uniform_sample!, 
           prioritized_sample!, capacity, normalize!, extra_columns, get_last_N_indices, 
           push_reservoir!, get_episodes
    include("experience_buffer.jl")
    
    export TrainingParams, batch_train!
    include("training.jl")
    
    export PolicyParams, trajectory_pdf, NetworkPolicy, polyak_average!, ContinuousNetwork, DiscreteNetwork, 
           DoubleNetwork, ActorCritic, GaussianPolicy, SquashedGaussianPolicy, 
           DistributionPolicy, MixedPolicy, MixtureNetwork, ϵGreedyPolicy,
           GaussianNoiseExplorationPolicy, FirstExplorePolicy, 
           entropy, logpdf, action_space, exploration, layers, actor, critic,
           LatentConditionedNetwork, StateDependentDistributionPolicy
    include("policies.jl")
    
    export Sampler, initial_observation, terminate_episode!, step!, steps!, 
           episodes!, metric_by_key, metrics_by_key, undiscounted_return, 
           discounted_return, failure, fill_gae!, fill_returns!, trim!
    include("sampler.jl")
    
    export elapsed, LoggerParams, aggregate_info, log_performance, 
           log_discounted_return, log_undiscounted_return, log_failure, 
           log_exploration, log_metric_by_key, log_metrics_by_key, log_validation_error,
           log_episode_averages, log_experience_sums, save_gif
    include("logging.jl")
    
    export smooth, readtb, tb2dict, plot_learning, episode_frames, gif, percentile, 
           find_crossing, plot_jumpstart, directories, plot_peak_performance, 
           plot_learning, plot_cumulative_rewards, plot_steps_to_threshold
    include("analysis.jl")
    
    export BatchAdjacentBuffer
    include("extras/batch_adjacent_buffer.jl")
    
    export DenseSN, ConvSN
    include("extras/spectral_normalization.jl")
    
    export GAN_BCELoss, GAN_LSLoss, GAN_HingeLoss, GAN_WLossGP, GAN_WLoss, Lᴰ, Lᴳ, GANLosses
    include("extras/gans.jl")
    
    export gradient_penalty
    include("extras/gradient_penalty.jl")
    
    export DiagonalFisherRegularizer, add_fisher_information_diagonal!, update_fisher!
    include("extras/fisher_information.jl")
    
    export BatchRegularizer, value_regularization, action_regularization, action_value_regularization
    include("extras/batch_regularization.jl")
    
    export OrthogonalRegularizer
    include("extras/orthogonal_regularization.jl")
    
    export log_multitask_performances!, continual_learning
    include("extras/multitask_learning.jl")
    
    export DeepEnsemble, DeepClassificationEnsemble, training_loss
    include("extras/deep_ensembles.jl")
    
    export cross_entropy, mcmc
    include("extras/bayesian_inference.jl")
    
    export OnPolicySolver, OffPolicySolver, BatchSolver
    include("model_free/on_policy.jl")
    include("model_free/off_policy.jl")
    include("model_free/batch.jl")
    
    export REINFORCE, A2C, PPO, LagrangePPO, DQN, DDPG, TD3, SAC
    include("model_free/rl/reinforce.jl")
    include("model_free/rl/a2c.jl")
    include("model_free/rl/ppo.jl")
    include("model_free/rl/dqn.jl")
    include("model_free/rl/ddpg.jl")
    include("model_free/rl/td3.jl")
    include("model_free/rl/sac.jl")
    
    export OnPolicyGAIL, OffPolicyGAIL, BC, AdVIL, SQIL, AdRIL, ASAF
    export mse_action_loss, logpdf_bc_loss, mse_value_loss, NDA_GAIL_JS
    include("model_free/il/bc.jl")
    include("model_free/il/AdVIL.jl")
    include("model_free/il/on_policy_gail.jl")
    include("model_free/il/off_policy_gail.jl")
    include("model_free/il/asaf.jl")
    include("model_free/il/sqil.jl")
    include("model_free/il/AdRIL.jl")
    include("model_free/il/nda_gail_js.jl")
    
    export CQL, BatchSAC
    include("model_free/batch/cql.jl")
    include("model_free/batch/sac.jl")
    
    export ExperienceReplay, TIER
    include("model_free/cl/experience_replay.jl")
    include("model_free/cl/tier.jl")
    
    export AdversarialOffPolicySolver, RARL, RARL_DQN, RARL_TD3, ISARL, ISARL_DQN, ISARL_TD3
    include("model_free/adversarial/adv_off_policy.jl")
    include("model_free/adversarial/rarl.jl")
    include("model_free/adversarial/isarl.jl")
end

