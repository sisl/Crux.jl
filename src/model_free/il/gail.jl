function GAIL(;D::ContinuousNetwork, solver, gan_loss::GANLoss, d_opt::NamedTuple=(;), 𝒟_expert, kwargs...)
    d_opt = TrainingParams(;loss = (D, 𝒟_ex, 𝒟_π; info = Dict()) -> Lᴰ(gan_loss, D, 𝒟_ex[:a], 𝒟_π[:a], wD = 𝒟_ex[:expert_val], yD = (𝒟_ex[:s],), yG = (𝒟_π[:s],)), d_opt...)
    function GAIL_callback(𝒟; info=Dict())
        info_D = batch_train!(D, d_opt, 𝒟_expert, 𝒟)
        merge!(info, info_D)
        
        discriminator_signal = haskey(𝒟, :advantage) ? :advantage : :return
        𝒟[discriminator_signal] .= whiten(value(D, 𝒟[:a], 𝒟[:s])) # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
    end
    𝒮 = solver(;post_batch_callback=GAIL_callback, kwargs...)
    𝒮.c_opt = nothing
    𝒮
end