function TIER_td_loss(;loss=Flux.mse, name=:Qavg, s_key=:s, a_key=:a, weight=nothing)
    (π, 𝒫, 𝒟, y; info=Dict(), z) -> begin
        Q = value(π, vcat(𝒟[s_key], z), 𝒟[a_key]) 
        
        # Store useful information
        Zygote.ignore() do
            info[name] = mean(Q)
        end
        
        loss(Q, y, agg = isnothing(weight) ? mean : weighted_mean(𝒟[weight]))
    end
end

function TIER_double_Q_loss(;name1=:Q1avg, name2=:Q2avg, kwargs...)
    l1 = TIER_td_loss(;name=name1, kwargs...)
    l2 = TIER_td_loss(;name=name2, kwargs...)
    
    (π, 𝒫, 𝒟, y; info=Dict(), z=𝒟[:z]) -> begin
        .5f0*(l1(critic(π).N1, 𝒫, 𝒟, y, info=info, z=z) + l2(critic(π).N2, 𝒫, 𝒟, y, info=info,z=z))
    end
end


TIER_TD3_actor_loss(π, 𝒫, 𝒟; info = Dict()) = -mean(value(critic(π).N1, vcat(𝒟[:s], 𝒟[:z]), action(actor(π), vcat(𝒟[:s], 𝒟[:z]))))

function TIER_TD3_target(π, 𝒫, 𝒟, γ::Float32; i, z=𝒟[:z]) 
    ap, _ = exploration(𝒫[:π_smooth], vcat(𝒟[:sp], z), π_on=actor(π), i=i)
    y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(critic(π), vcat(𝒟[:sp], z), ap)...)
end

TIER_action_regularization(π, 𝒟) = Flux.mse(action(actor(π), vcat(𝒟[:s], 𝒟[:z])), 𝒟[:a])
TIER_action_value_regularization(π, 𝒟) = begin 
    v = value(critic(π), vcat(𝒟[:s], 𝒟[:z]), 𝒟[:a])
    v isa Tuple && (v = v[1])
    Flux.mse(v, 𝒟[:value])
end


function TIER(;π, 
               observation_model,
               S, 
               ΔN,
               𝒫=(;),
               A=action_space(π), 
               buffer_size=1000, 
               latent_dim, 
               N_experience_replay, 
               N_experience_obs,
               ER_frac = 0.5,
               replay_store_weight = (D)->1f0,
               bayesian_inference,
               solver=TD3, 
               required_columns=Symbol[], 
               c_opt=(;), 
               a_opt=(;), 
               obs_opt=(;),
               a_loss=TIER_TD3_actor_loss,
               c_loss=TIER_double_Q_loss(),
               target_fn=TIER_TD3_target,
               zprior=MvNormal(zeros(latent_dim), I),
               kwargs...)
    # if args["bayesian_inference"]=="mcmc"
   	# 	z_dist = rand(z_dist, args["N_BI_samples"]) # prior for mcmc
    required_columns = unique([:weight, required_columns...])
    buffer = ExperienceBuffer(S, A, buffer_size, required_columns)
    buffer.data[:z] = zeros(Float32, latent_dim, capacity(buffer))
      
    # This experience buffer is for recalling behavior
    buffer_er = ExperienceBuffer(S, A, N_experience_replay, [required_columns..., :value])
    buffer_er.data[:z] = zeros(Float32, latent_dim, capacity(buffer_er))

    # this experience buffer is for learning the latent embedding
    buffer_obs = ExperienceBuffer(S, A, N_experience_obs, [required_columns..., :value])
    buffer_obs.data[:z] = zeros(Float32, latent_dim, capacity(buffer_obs))

    # Buffer used to train for latent
    obs_opt = TrainingParams(;obs_opt...)
    𝒟obs = buffer_like(buffer_obs, capacity=obs_opt.batch_size)
    
    𝒫 = (;buffer_er, buffer_obs, obs_opt, z_dist=Any[zprior], zs=Any[zprior], observation_model, 𝒫...)

    # Define the solver 
    𝒮 = solver(;π=π,
              S=S,
              𝒫=𝒫,
              ΔN=ΔN,
              buffer=buffer,
              required_columns=required_columns,
              c_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=TIER_action_value_regularization), c_opt...),
              a_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=TIER_action_regularization), a_opt...),
              extra_buffers = [buffer_er],
              buffer_fractions = [1.0 - ER_frac, ER_frac],
              a_loss=a_loss,
              c_loss=c_loss,
              target_fn=target_fn,
              kwargs...
              )

    function TIER_cb(D; 𝒮, info=Dict())
        # update the z distribution
        𝒮.𝒫.z_dist[1] = bayesian_inference(observation_model, D, 𝒮.𝒫.z_dist[1], info=info)
        
        # Set the agent's best estimate and record
        zbest = Crux.best_estimate(𝒮.𝒫.z_dist[1])
        𝒮.agent.π.z = zbest
        push!(𝒮.𝒫[:zs], deepcopy(𝒮.𝒫.z_dist[1]))
        
        # Fill the buffer with latent estimate, value and computed weight
        D[:z] = repeat(zbest, 1, length(D[:r]))
        D[:value] = mean(value(critic(𝒮.agent.π), vcat(D[:s], D[:z]), D[:a]))
        D[:weight] .= replay_store_weight(D)
        
        # Add this buffer to our experience replay and observation buffers
        push_reservoir!(buffer_er, D, weighted=true)
        push_reservoir!(buffer_obs, D)
        
        # Train the obs model
        for j=1:obs_opt.epochs
                rand!(𝒟obs, buffer_obs, buffer, fracs=[0.5, 0.5])
                train!(Flux.params(observation_model), (;kwargs...) -> obs_opt.loss(observation_model, 𝒟obs; kwargs...), obs_opt, info=info)
        end
    end

    𝒮.post_sample_callback = TIER_cb
    𝒮
end

