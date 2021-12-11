function NDA_GAIL_JS(;π, 
                       S, 
                       𝒟_demo,
                       𝒟_nda,
                       normalize_demo::Bool=true, 
                       D::ContinuousNetwork,
                       Dnda::ContinuousNetwork,
                       solver=PPO, 
                       gan_loss::Crux.GANLoss=GAN_BCELoss(), 
                       d_opt::NamedTuple=(;),
                       d_opt_nda::NamedTuple=(;), 
                       log::NamedTuple=(;),  
                       kwargs...)
                       
    d_opt = TrainingParams(;loss = Crux.GAIL_D_loss(gan_loss), name="discriminator_", d_opt...)
    d_opt_nda = TrainingParams(;loss = Crux.GAIL_D_loss(gan_loss), name="nda_discriminator_", d_opt_nda...)
    if normalize_demo
        𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π))
        𝒟_nda = normalize!(deepcopy(𝒟_nda), S, action_space(π))
    end
    𝒟_demo = 𝒟_demo |> device(π)
    𝒟_nda = 𝒟_nda |> device(π)
    
    function GAIL_callback(𝒟; info=Dict())
        batch_train!(D, d_opt, (;), 𝒟_demo, 𝒟, info=info)
        batch_train!(Dnda, d_opt_nda, (;), 𝒟_nda, 𝒟, info=info)
        
        # Set the reward
        discriminator_signal = haskey(𝒟, :advantage) ? :advantage : :return
        D_out = value(D, 𝒟[:a], 𝒟[:s]) # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
        r = Base.log.(sigmoid.(D_out) .+ 1f-5) .- Base.log.(1f0 .- sigmoid.(D_out) .+ 1f-5)
        ignore() do
            minval, maxval = extrema(D_out)
            println("minval: $minval, maxval: $maxval")
            info["disc_reward"] = mean(r)
        end
        𝒟[discriminator_signal] .= r 
        
        # Set the cost
        D_out_nda = value(Dnda, 𝒟[:a], 𝒟[:s])
        r_nda = Base.log.(sigmoid.(D_out_nda) .+ 1f-5) .- Base.log.(1f0 .- sigmoid.(D_out_nda) .+ 1f-5)
        c = max.(0, r_nda .- r)
        ignore() do
            info["disc_nda_cost"] = sum(c) / sum(𝒟[:episode_end])
        end
        𝒟[:cost] .= c
        
    end
    𝒮 = solver(;π=π, S=S, post_batch_callback=GAIL_callback, log=(dir="log/onpolicygail", period=500, log...), kwargs...)
    𝒮.c_opt = nothing # disable the critic 
    𝒮
end

