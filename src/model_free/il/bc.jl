mse_action_loss(π, 𝒫, 𝒟; kwargs...) = Flux.mse(action(π, 𝒟[:s]), 𝒟[:a])

function mse_value_loss(π, 𝒫, 𝒟; kwargs...)
    eloss = -mean(entropy(π, 𝒟[:s]))
    mseloss = Flux.mse(value(π, 𝒟[:s]), 𝒟[:value])
    𝒫[:λe]*eloss + mseloss
end

function logpdf_bc_loss(π, 𝒫, 𝒟; info=Dict())
    eloss = -mean(entropy(π, 𝒟[:s]))
    lloss = -mean(logpdf(π, 𝒟[:s], 𝒟[:a]))
    ignore_derivatives() do
        info[:entropy] = -eloss
        info[:logpdf] = lloss
    end
    𝒫[:λe]*eloss + lloss
end


"""
Behavioral cloning solver.

```julia
BC(;
    π,
    S,
    𝒟_demo,
    normalize_demo::Bool=true,
    loss=nothing,
    validation_fraction=0.3,
    window=100,
    λe::Float32=1f-3,
    opt::NamedTuple=(;),
    log::NamedTuple=(;),
    kwargs...)
```
"""
function BC(;
        π,
        S,
        𝒟_demo,
        normalize_demo::Bool=true,
        loss=nothing,
        validation_fraction=0.3,
        window=100,
        λe::Float32=1f-3,
        opt::NamedTuple=(;),
        log::NamedTuple=(;),
        kwargs...)

    if isnothing(loss)
        loss = π isa ContinuousNetwork ? mse_action_loss : logpdf_bc_loss
    end
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π)))
    𝒟_demo = 𝒟_demo |> device(π)

    # Split between train and validation sets
    shuffle!(𝒟_demo)
    𝒟_train, 𝒟_validate = split(𝒟_demo, [1-validation_fraction, validation_fraction])
    #TODO: We should include a validation loss, then early stopping should just analyze the history of the validation loss.

    𝒫 = (λe=λe,)
    BatchSolver(;agent=PolicyParams(π),
                 S=S,
                 𝒫=𝒫,
                 𝒟_train=𝒟_train,
                 a_opt=TrainingParams(;early_stopping=stop_on_validation_increase(π, 𝒫, 𝒟_validate, loss, window=window), loss=loss, opt...),
                 log=LoggerParams(;dir="log/bc", period=1, log...),
                 kwargs...)
end
