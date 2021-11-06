# ## Network for representing continous functions (value or policy)
# mutable struct LatentConditionedNetwork <: NetworkPolicy
#     policy
#     z
# end
# 
# device(π::LatentConditionedNetwork) = device(π.policy)
# 
# Flux.@functor LatentConditionedNetwork 
# 
# Flux.trainable(π::LatentConditionedNetwork) = Flux.trainable(π.policy)
# 
# layers(π::LatentConditionedNetwork) = layers(π.policy)
# 
# POMDPs.action(π::LatentConditionedNetwork, s; z=π.z) = value(π, s; z=z)
# 
# function POMDPs.value(π::LatentConditionedNetwork, s; z=π.z) 
#     if size(z, 2) != size(s)[end]
#         z = repeat(z, 1, ndims(s) == 1 ? 1 : size(s)[end])
#     end
#     value(π.policy, vcat(z,s))
# end
# 
# function POMDPs.value(π::LatentConditionedNetwork, s, a; z=π.z) 
#     if size(z, 2) != size(s)[end]
#         z = repeat(z, 1, ndims(s) == 1 ? 1 : size(s)[end])
#     end
#     value(π.policy, vcat(z,s), a)
# end
# 
# 
# action_space(π::LatentConditionedNetwork) = action_space(π.policy)
# 
# 
# @with_kw mutable struct OffPolicyLatentSolver <: Solver
#     π # Policy
#     S::AbstractSpace # State space
#     A::AbstractSpace = action_space(π) # Action space
#     N::Int = 1000 # Number of environment interactions
#     ΔN::Int = 4 # Number of interactions between updates
#     max_steps::Int = 100 # Maximum number of steps per episode
#     log::Union{Nothing, LoggerParams} = nothing # The logging parameters
#     i::Int = 0 # The current number of environment interactions
#     a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
#     c_opt::TrainingParams # Training parameters for the critic
#     post_batch_callback = (𝒟) -> nothing
# 
#     # Off-policy-specific parameters
#     π⁻ = deepcopy(π)
#     π_explore::Policy # exploration noise
#     target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
#     target_fn # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
#     buffer_size = 1000 # Size of the buffer
#     required_columns = Symbol[]
#     buffer::ExperienceBuffer = ExperienceBuffer(S, A, buffer_size, required_columns) # The replay buffer
#     buffer_init::Int = max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
#     extra_buffers = [] # extra buffers (i.e. for experience replay in continual learning)
#     buffer_fractions = [1.0] # Fraction of the minibatch devoted to each buffer
#     z_dists = MvNormal[]
# end
# 
# TIER(;π::ActorCritic, ΔN=50, π_explore=GaussianNoiseExplorationPolicy(0.1f0),  a_opt::NamedTuple=(;), c_opt::NamedTuple=(;), log::NamedTuple=(;), π_smooth::Policy=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), kwargs...) = 
#     OffPolicyLatentSolver(;
#         π=π, 
#         ΔN=ΔN,
#         log=LoggerParams(;dir = "log/ddpg", log...),
#         a_opt=TrainingParams(;loss=TD3_actor_loss_w_latent, name="actor_", a_opt...),
#         c_opt=TrainingParams(;loss=double_Q_loss_w_latent, name="critic_", epochs=ΔN, c_opt...),
#         π_explore=π_explore,
#         target_fn=TD3_target_w_latent(π_smooth),
#         kwargs...)
# 
# # function optimize_latent(loss, latent_dim)
# #     rng = MersenneTwister(0)
# #     z_prospective = [Float32.(rand(rng, Uniform(-1,1), latent_dim)) for i=1:100]
# #     vals = [loss(z) for z in z_prospective]
# #     z_prospective[argmin(vals)]
# # end
# 
# function cross_entropy_optimization(f, P, latent_dim)
#     m = max(floor(Int, 1000 * norm(P.Σ)), 5*(latent_dim+3))
#     m_elite = floor(Int, 0.2*m)
#     # P = MvNormal(zeros(latent_dim), 0.5*I)
#     for i=1:1
#     # mi = max(floor(Int, m / length(Ps)), 1)
#     # samples = clamp.(hcat([rand(P,mi) for P in Ps]...), -1f0, 1f0)
#         # P = MvNormal(P.μ, P.Σ + 1f-5*I)
#         samples = clamp.(rand(P, m), -1f0, 1f0)
#         samples .+= rand(Uniform(-1f-5, 1f-5), size(samples)...)
#         order = sortperm([f(samples[:,i]) for i=1:m])
#         P = fit(MvNormal, Float64.(samples[:, order[1:m_elite]]))
#     end
#     P
# end
# 
# function find_latent!(loss, P, latent_dim)
#     # 𝒟[:μ_z] .= optimize_latent(loss, latent_dim)
#     # Ps = [MvNormal(𝒟[:μ_z][:,i], 𝒟[:Σ_z][:,:,i] + 1f-5 * I ) for i=1:length(𝒟)]
#     P = cross_entropy_optimization(loss, P, latent_dim)
#     # 𝒟[:μ_z] .= P.μ
#     # P
#     # 𝒟[:Σ_z] .= P.Σ    
# end
# 
# function latent_loss(π⁻, π, γ, 𝒟)
#     (z) -> begin
#         y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(critic(π⁻), 𝒟[:sp], action(actor(π⁻), 𝒟[:sp], z=z), z=z)
#         Q = value(critic(π), 𝒟[:s], 𝒟[:a], z=z)
#         Flux.mae(Q, y)
#     end
# end
# 
# function TD3_latent_loss(π⁻, π, γ, 𝒟)
#     (z) -> begin
#         y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(critic(π).N1, 𝒟[:sp], action(actor(π), 𝒟[:sp], z=z), z=z)
#         Q = value(critic(π).N1, 𝒟[:s], 𝒟[:a], z=z)
#         Flux.mae(Q, y)
#     end
# end
# 
# function latent_target(π, 𝒟, γ; z=𝒟[:μ_z], kwargs...)
#     𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(critic(π), 𝒟[:sp], action(actor(π), 𝒟[:sp], z=z), z=z)
# end 
# 
# function critic_loss_w_latent(π, 𝒟, y; loss=Flux.mae, weighted=false, name=:Qavg, info=Dict())
#     Q = value(critic(π), 𝒟[:s], 𝒟[:a]; z=𝒟[:μ_z]) 
# 
# 
#     # Store useful information
#     ignore() do
#         info[name] = mean(Q)
#     end
# 
#     loss(Q, y, agg = weighted ? weighted_mean(𝒟[:weight]) : mean)
# end
# 
# function double_Q_loss_w_latent(π, 𝒟, y; info=Dict(), weighted=false)
#     q1loss = critic_loss_w_latent(π.C.N1, 𝒟, y, info=info, name=:Q1avg, weighted=weighted)
#     q2loss = critic_loss_w_latent(π.C.N2, 𝒟, y, info=info, name=:Q2avg, weighted=weighted)
#     q1loss + q2loss
# end
# 
# function actor_loss_w_latent(π, 𝒟; info=Dict()) 
#     -mean(value(critic(π), 𝒟[:s], action(actor(π), 𝒟[:s], z=𝒟[:μ_z]), z=𝒟[:μ_z]))
# end
# 
# TD3_actor_loss_w_latent(π, 𝒟; info = Dict()) = -mean(value(critic(π).N1, 𝒟[:s], action(actor(π), 𝒟[:s],z=𝒟[:μ_z]), z=𝒟[:μ_z]))
# 
# function TD3_target_w_latent(π_smooth)
#     (π, 𝒟, γ::Float32; i,  z=𝒟[:μ_z]) -> begin
#         ap, _ = exploration(π_smooth, 𝒟[:sp], π_on=π, i=i)
#         y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(critic(π).N1, 𝒟[:sp], ap, z=z), value(critic(π).N2, 𝒟[:sp], ap, z=z))
#     end
# end
# 
# action_regularization_tier(π, 𝒟s) = length(𝒟s) == 0 ? 0 : mean([Flux.mse(action(actor(π), 𝒟[:s], z=𝒟[:μ_z]), 𝒟[:a]) for 𝒟 in 𝒟s])
# action_value_regularization_tier(π, 𝒟s) = length(𝒟s) == 0 ? 0 : mean([Flux.mse(value(critic(π).N1, 𝒟[:s], 𝒟[:a], z=𝒟[:μ_z]), 𝒟[:value]) for 𝒟 in 𝒟s]) +  mean([Flux.mse(value(critic(π).N2, 𝒟[:s], 𝒟[:a], z=𝒟[:μ_z]), 𝒟[:value]) for 𝒟 in 𝒟s])
# 
# 
# 
# function POMDPs.solve(𝒮::OffPolicyLatentSolver, mdp)
#     # Compute the latent dimension
#     latent_dim = length(actor(𝒮.π).z)
# 
#     # Add data for normal distributions
#     # C = capacity(𝒮.buffer)
#     # 𝒮.buffer.data[:μ_z] = zeros(Float32, latent_dim, C)
#     # 𝒮.buffer.data[:Σ_z] = 0.5f0*repeat(Array{Float32, 2}(I, latent_dim, latent_dim), outer=[1,1,C])
# 
#     # Create minibatch buffers for each buffer
#     allbuffs = [𝒮.extra_buffers..., 𝒮.buffer]
#     push!(𝒮.z_dists, MvNormal(zeros(latent_dim), 0.5*I))
#     # 
#     # for b in allbuffs
#     #     @assert haskey(b, :μ_z) && haskey(b, :Σ_z)
#     # end
# 
# 
# 
#     batches = split_batches(𝒮.c_opt.batch_size, 𝒮.buffer_fractions)
#     𝒟s = [buffer_like(b, capacity=batchsize, device=device(𝒮.π)) for (b, batchsize) in zip(allbuffs, batches)]
#     # Add latent dimension
#     # last_z = [zeros(Float32, latent_dim) for _ in 𝒟s]
# 
#     for 𝒟 in 𝒟s
#         𝒟.data[:μ_z] = zeros(Float32, latent_dim, capacity(𝒟))
#         # 𝒟.data[:Σ_z] = 0.5f0*repeat(Array{Float32, 2}(I, latent_dim, latent_dim), outer=[1,1,capacity(𝒟)])
#         𝒟.data[:value] = zeros(Float32, 1, capacity(𝒟))
#     end
# 
#     γ = Float32(discount(mdp))
#     s = Sampler(mdp, 𝒮.π, S=𝒮.S, A=𝒮.A, max_steps=𝒮.max_steps, π_explore=𝒮.π_explore, required_columns=extra_columns(𝒮.buffer))
#     isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)
# 
#     # Log the pre-train performance
#     log(𝒮.log, 𝒮.i)
# 
#     # Fill the buffer with initial observations before training
#     𝒮.i += fillto!(𝒮.buffer, s, 𝒮.buffer_init, i=𝒮.i, explore=true)
# 
#     # Loop over the desired number of environment interactions
#     for 𝒮.i in range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
#         # Sample transitions into the replay buffer
#         data = steps!(s, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i)
#         # data[:μ_z] = repeat(actor(𝒮.π).z, outer=[1,𝒮.ΔN])
#         # data[:Σ_z] = repeat(Array{Float32, 2}(I, latent_dim, latent_dim), outer=[1,1,𝒮.ΔN])
#         push!(𝒮.buffer, data)
# 
#         # callback for potentially updating the buffer
#         𝒮.post_batch_callback(𝒮.buffer) 
# 
#         # Determine the latent variables
#         info_z = Dict()
#         for (b, index, Dz) in zip(allbuffs, 1:length(𝒟s), 𝒮.z_dists)
#             D = minibatch(b, rand(1:length(b), 1000))
#             z_before = 𝒮.z_dists[index].μ
#             𝒮.z_dists[index] = find_latent!(TD3_latent_loss(𝒮.π⁻, 𝒮.π, γ, D), Dz, latent_dim)
#             # b[:μ_z][:,𝒟.indices] .= 𝒟[:μ_z]
#             # b[:Σ_z][:,:,𝒟.indices] .= 𝒟[:Σ_z]
#             z_after = 𝒮.z_dists[index].μ
#             info_z["Δz$index"] = norm(z_after .- z_before) # Store the change in task identifier
#             info_z["z$index"] = norm(z_after) # Store the change in task identifier
#             info_z["Σ$index"] = norm(𝒮.z_dists[index].Σ) # Store the norm of the covariance
#             info_z["td_error$index"] = mean(abs.(value(critic(𝒮.π).N1, D[:s], D[:a], z=z_after)  .- 𝒮.target_fn(𝒮.π⁻, D, γ, i=𝒮.i, z=z_after))) # store td error on the ith task
#         end
# 
# 
#         infos = []
#         # Loop over the desired number of training steps
#         for epoch in 1:𝒮.c_opt.epochs
#             # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
#             for (𝒟, b, index) in zip(𝒟s, allbuffs, 1:length(𝒟s))
#                 rand!(𝒟, b, i=𝒮.i) # sample a batch
#                 𝒟[:μ_z] .= 𝒮.z_dists[index].μ # set the parameters
#             end
# 
#             # Set the latent variable of the current actor and critic
#             z = 𝒮.z_dists[end].μ
#             actor(𝒮.π).z .= z
#             critic(𝒮.π).N1.z .= z
#             critic(𝒮.π).N2.z .= z
# 
#             # concatenate the minibatch buffers
#             𝒟 = hcat(𝒟s...)
# 
#             # Compute target
#             y = 𝒮.target_fn(𝒮.π⁻, 𝒟, γ, i=𝒮.i)
# 
#             # Train the critic
#             info = train!(critic(𝒮.π), (;kwargs...) -> 𝒮.c_opt.loss(𝒮.π, 𝒟, y; kwargs...) + action_value_regularization_tier(𝒮.π, 𝒟s[1:end-1]), 𝒮.c_opt)
# 
#             # Train the actor 
#             if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
#                 info_a = train!(actor(𝒮.π), (;kwargs...) -> 𝒮.a_opt.loss(𝒮.π, 𝒟; kwargs...) + action_regularization_tier(𝒮.π, 𝒟s[1:end-1]), 𝒮.a_opt)
#                 info = merge(info, info_a)
# 
#                 # Update the target network
#                 𝒮.target_update(𝒮.π⁻, 𝒮.π)
#             end
# 
#             # Store the training information
#             push!(infos, info)
# 
#         end
#         # If not using a separate actor, update target networks after critic training
#         isnothing(𝒮.a_opt) && 𝒮.target_update(𝒮.π⁻, 𝒮.π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
# 
#         # Log the results
#         log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, aggregate_info(infos), info_z)
#     end
#     𝒮.i += 𝒮.ΔN
#     𝒮.π
# end
# 
# 
# # function optimize_latent(π, loss, i, kmax=10)
# #     z, y = π.z, loss()
# #     z_best, y_best = π.z, y
# #     scale = Float32((1/5000)*i + 1)
# #     for k in 1:kmax
# #         π.z = min.(max.(z .+ Float32.(randn(size(z)...)) ./ scale, -1f0), 1f0)
# #         y′ = loss()
# #         Δy = y′ - y
# #         if Δy ≤ 0 || rand() < exp(-Δy*i)
# #             z, y = π.z, y′
# #         end
# #         if y′ < y_best
# #             z_best, y_best = π.z, y′
# #         end
# #     end
# #     π.z = z_best 
# # end
# 
