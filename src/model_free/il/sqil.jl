function sqil_callback(𝒟; kwargs...)
    𝒟[:r] .= 0
end

"""
Soft Q Imitation Learning (SQIL) solver.

```julia
SQIL(;
    π, 
    S, 
    𝒟_demo, 
    normalize_demo::Bool=true, 
    solver=SAC, 
    log::NamedTuple=(;), 
    kwargs...)
```
"""
function SQIL(;
        π, 
        S, 
        𝒟_demo, 
        normalize_demo::Bool=true, 
        solver=SAC, 
        log::NamedTuple=(;), 
        kwargs...)
               
    !haskey(𝒟_demo, :r) && error("SQIL requires a reward value for the demonstrations")
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)
    solver(;π=π, 
            S=S, 
            post_sample_callback=sqil_callback, 
            extra_buffers=[𝒟_demo],
            buffer_fractions=[1/2, 1/2],
            log=(dir="log/sqil", period=500, log...),
            kwargs...)
end

