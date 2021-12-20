function ASAF_actor_loss(πG, 𝒟_demo)
    (π, 𝒫, 𝒟,; info=Dict()) -> begin
        πsa_G = logpdf(π, 𝒟[:s], 𝒟[:a])
        πsa_E = logpdf(π, 𝒟_demo[:s], 𝒟_demo[:a])
        πGsa_G = logpdf(πG, 𝒟[:s], 𝒟[:a])
        πGsa_E = logpdf(πG, 𝒟_demo[:s], 𝒟_demo[:a])
        e = mean(entropy(π, 𝒟[:s]))
        
        ignore() do
            info[:entropy] = e
        end 
        
        L = Flux.mean(log.(1 .+ exp.(πGsa_E - πsa_E))) + Flux.mean(log.(exp.(πsa_G - πGsa_G)  .+ 1))  - 0.1f0*e
        # if !isnothing(𝒟_nda)
        #     πsa_NDA = logpdf(π, 𝒟_nda[:s], 𝒟_nda[:a])
        #     πGsa_NDA = logpdf(πG, 𝒟_nda[:s], 𝒟_nda[:a])
        #     L += Flux.mean(log.(1 .+ exp.(πsa_NDA - πGsa_NDA)))
        # end
        L
    end
end


function ASAF(;π, 
               S, 
               𝒟_demo, 
               normalize_demo::Bool=true, 
               ΔN=50, 
               λ_orth=1f-4, 
               a_opt::NamedTuple=(;), 
               c_opt::NamedTuple=(;), 
               log::NamedTuple=(;), 
               kwargs...)
               
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)
    OnPolicySolver(;agent=PolicyParams(π), 
                    S=S,
                    ΔN=ΔN,
                    post_batch_callback=(D; 𝒮, kwargs...) -> 𝒮.a_opt.loss = ASAF_actor_loss(deepcopy(𝒮.agent.π), 𝒟_demo),
                    log=LoggerParams(;dir="log/ASAF", period=100, log...),
                    a_opt=TrainingParams(;name="actor_", loss=nothing, a_opt...), 
                    kwargs...)
end

