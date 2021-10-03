function AdRIL(;π, 
                S,
                ΔN=50,
                solver=SAC, 
                𝒟_demo, 
                normalize_demo::Bool=true,
                expert_frac=0.5, 
                buffer_size = 1000, 
                buffer_init=0,
                log::NamedTuple=(;),
                buffer::ExperienceBuffer = ExperienceBuffer(S, action_space(π), buffer_size, [:i]), 
                kwargs...)
    
    !haskey(𝒟_demo, :r) && error("AdRIL requires a reward value for the demonstrations")
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)
    
    
    function AdRIL_callback(𝒟)
        max_i = maximum(𝒟[:i])
        k = Int((max_i - buffer_init) / ΔN) - 1
        old_data = 𝒟[:i][:] .<= max_i - ΔN
        new_data = 𝒟[:i][:] .> max_i - ΔN
        𝒟[:r][1, old_data] .= -1/k
        𝒟[:r][1, new_data] .= 0
    end
    
    
    solver(;π=π, 
            S=S, 
            ΔN=ΔN, 
            post_experience_callback=AdRIL_callback, 
            extra_buffers=[𝒟_demo], 
            buffer_fractions=[1-expert_frac, expert_frac], 
            buffer_size=buffer_size,
            buffer_init=buffer_init, 
            buffer=buffer, 
            log=(dir="log/adril", period=500, log...),
            kwargs...)
end

