function ExperienceReplay(;π, 
                          S, 
                          𝒫=(;),
                          A=action_space(π), 
                          N_experience_replay, 
                          solver=TD3, 
                          required_columns=Symbol[],  
                          c_opt=(;), 
                          a_opt=(;), 
                          replay_store_weight = (D)->1f0,
                          kwargs...)
                          
    buffer_er = ExperienceBuffer(S, A, N_experience_replay, [:episode_end, :weight, :value, required_columns...])
    𝒫 = (;buffer_er, 𝒫...)
    function cb(D; 𝒮, info=Dict())
        D[:value] = mean(value(𝒮.agent.π, D[:s], D[:a]))
        D[:weight] .= replay_store_weight(D)
        push_reservoir!(buffer_er, D, weighted=true)
    end
    
    solver(;  π=π,
              S=S,
              𝒫=𝒫,
              required_columns=[:episode_end, :weight, required_columns...],
              post_sample_callback=cb,
              c_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=action_value_regularization, c_opt...)),
              a_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=action_regularization, a_opt...)),
              extra_buffers=[buffer_er],
              buffer_fractions=[0.5, 0.5],
              kwargs...
              )
end

