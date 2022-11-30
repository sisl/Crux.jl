function GAIL_D_loss(gan_loss)
    (D, 𝒫, 𝒟_ex, 𝒟_π; info = Dict()) ->begin
        Lᴰ(gan_loss, D, 𝒟_ex[:a], 𝒟_π[:a], yD = (𝒟_ex[:s],), yG = (𝒟_π[:s],))
    end
end

function OnPolicyGAIL(;π,
                       S,
                       γ,
                       λ_gae::Float32 = 0.95f0,
                       𝒟_demo,
                       αr::Float32 = 0.5f0,
                       normalize_demo::Bool=true,
                       D::ContinuousNetwork,
                       solver=PPO,
                       gan_loss::GANLoss=GAN_BCELoss(),
                       d_opt::NamedTuple=(;),
                       log::NamedTuple=(;),
                       Rscale=1f0,
                       kwargs...)

    d_opt = TrainingParams(;loss = GAIL_D_loss(gan_loss), name="discriminator_", d_opt...)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)

    function GAIL_callback(𝒟; info=Dict(), 𝒮)
        batch_train!(D, d_opt, (;), 𝒟_demo, deepcopy(𝒟), info=info)

        D_out = value(D, 𝒟[:a], 𝒟[:s]) # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
        r = αr * logσ.(D_out) .- (1f0 - αr) * logcompσ.(D_out)
        ignore_derivatives() do
            info["disc_reward"] = mean(r)
        end

        𝒟[:r] .= r.*Rscale

        eps = episodes(𝒟)
        for ep in eps
            eprange = ep[1]:ep[2]
            fill_gae!(𝒟, eprange, 𝒮.agent.π, λ_gae, γ)
            fill_returns!(𝒟, eprange, γ)
        end
        𝒟[:advantage] .= whiten(𝒟[:advantage])

    end
    solver(;π=π, S=S, post_batch_callback=GAIL_callback, log=(dir="log/onpolicygail", period=500, log...), λ_gae=λ_gae, kwargs...)
end
