function RARL_DQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
         -𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(value(π, 𝒟[:sp]), dims=1)
end

function RARL_TD3_target(π, 𝒫, 𝒟, γ::Float32; i) 
    ap, _ = exploration(𝒫[:π_smooth], 𝒟[:sp], π_on=π, i=i)
    y = -𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π, 𝒟[:sp], ap)...)
end

RARL_DQN(;kwargs...) = DQN(;c_loss=td_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=RARL_DQN_target)
RARL_TD3(;kwargs...) = DQN(;c_loss=double_Q_loss(name=:x_Qavg, a_key=:x), prefix="x_", target_fn=RARL_TD3_target)

function RARL(;𝒮_pro,
               𝒮_ant,
               px,
               log::NamedTuple=(;), 
               train_pro_every::Int=1,
               train_ant_every::Int=1,
               buffer_size=1000, # Size of the buffer
               required_columns=Symbol[:x, :fail],
               buffer::ExperienceBuffer=ExperienceBuffer(𝒮_pro.S, 𝒮_pro.agent.space, buffer_size, required_columns), # The replay buffer
               buffer_init::Int=max(max(𝒮_pro.c_opt.batch_size, 𝒮_ant.c_opt.batch_size), 200) # Number of observations to initialize the buffer with
               )
        # Set buffer parameters to be consistent between solvers (Since buffer is shared)
        𝒮_pro.required_columns = required_columns
        𝒮_pro.buffer = buffer
        𝒮_pro.buffer_init = buffer_init
        𝒮_ant.required_columns = required_columns
        𝒮_ant.buffer = buffer    
        𝒮_ant.buffer_init = buffer_init    
        AdversarialOffPolicySolver(;𝒮_pro=𝒮_pro,
                                    𝒮_ant=𝒮_ant,
                                    px=px,
                                    train_pro_every=train_pro_every,
                                    train_ant_every=train_ant_every,
                                    log=LoggerParams(;dir="log/rarl", log...),)
end
