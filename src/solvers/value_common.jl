# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)

target(Q, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q(𝒟[:sp]), dims=1)

q_predicted(π::Policy, 𝒟) = sum(value(π, 𝒟[:s]) .* 𝒟[:a], dims = 1)
q_predicted(Q::Chain, 𝒟) = sum(Q(𝒟[:s]) .* 𝒟[:a], dims = 1)

td_loss(π, 𝒟, y, L) =  L(q_predicted(π, 𝒟), y, agg = weighted_mean(𝒟[:weight]))

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)
