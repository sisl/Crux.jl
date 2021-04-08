mse_action_loss() = (π, 𝒟; kwargs...) -> Flux.mse(action(π, 𝒟[:s]), 𝒟[:a])
function mse_value_loss(λe::Float32)
    (π, 𝒟; kwargs...) -> begin
        eloss = -mean(entropy(π, 𝒟[:s]))
        mseloss = Flux.mse(value(π, 𝒟[:s]), 𝒟[:value])
        λe*eloss + mseloss
    end
end
function logpdf_bc_loss(λe::Float32)
    (π, 𝒟; info=Dict())->begin
        eloss = -mean(entropy(π, 𝒟[:s]))
        lloss = -mean(logpdf(π, 𝒟[:s], 𝒟[:a]))
        ignore() do
            info[:entropy] = -eloss
            info[:logpdf] = lloss
        end
        λe*eloss + lloss
    end
end

function BC(;π, 𝒟_expert, S, A=action_space(π), loss=nothing, validation_fraction=0.3, window=100, λe::Float32=1f-3, opt::NamedTuple=(;), log::NamedTuple=(;), kwargs...)
    if isnothing(loss)
        loss = π isa ContinuousNetwork ? mse_action_loss() : logpdf_bc_loss(λe)
    end
    𝒟_expert = normalize!(deepcopy(𝒟_expert), S, A) |> device(π)
    shuffle!(𝒟_expert)
    𝒟_train, 𝒟_validate = split(𝒟_expert, [1-validation_fraction, validation_fraction])
    BatchSolver(;π=π, 
              S=S,
              A=A,
              𝒟_train=𝒟_train, 
              a_opt=TrainingParams(;early_stopping=stop_on_validation_increase(π, 𝒟_validate, loss, window=window), loss=loss, opt...), 
              log=LoggerParams(;dir="log/bc", period=1, log...),
              kwargs...)
end

