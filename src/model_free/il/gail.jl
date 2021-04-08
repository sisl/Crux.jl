function GAIL_D_loss(gan_loss)
    (D, 𝒟_ex, 𝒟_π; info = Dict()) -> begin
        Lᴰ(gan_loss, D, 𝒟_ex[:a], 𝒟_π[:a], yD = (𝒟_ex[:s],), yG = (𝒟_π[:s],))
    end
end

function GAIL(;π, S, A=action_space(π), D::ContinuousNetwork, solver=PPO, gan_loss::GANLoss=GAN_BCELoss(), d_opt::NamedTuple=(;), 𝒟_expert, normalization=Dict(), kwargs...)
    d_opt = TrainingParams(;loss = GAIL_D_loss(gan_loss), name="discriminator_", d_opt...)
    𝒟_expert_norm = normalize!(deepcopy(𝒟_expert), S, A) |> device(D)
    function GAIL_callback(𝒟; info=Dict())
        info_D = batch_train!(D, d_opt, 𝒟_expert_norm, 𝒟)
        merge!(info, info_D)
        
        discriminator_signal = haskey(𝒟, :advantage) ? :advantage : :return
        𝒟[discriminator_signal] .= whiten(value(D, 𝒟[:a], 𝒟[:s])) # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
    end
    𝒮 = solver(;π=π, S=S, A=A, post_batch_callback=GAIL_callback, kwargs...)
    𝒮.c_opt = nothing # disable the critic 
    𝒮
end

