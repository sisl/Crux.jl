function SQIL_callback(𝒟)
    𝒟[:r] .= 0
end

function SQIL(;π, S, A=action_space(π), 𝒟_demo, normalize_demo::Bool=true, solver=SAC, log::NamedTuple=(;), kwargs...)
    !haskey(𝒟_demo, :r) && error("SQIL requires a reward value for the demonstrations")
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> device(π)
    solver(;π=π, 
            S=S, 
            A=A, 
            post_batch_callback=SQIL_callback, 
            extra_buffers=[𝒟_demo],
            log=(dir="log/sqil", period=500, log...),
            kwargs...)
end

