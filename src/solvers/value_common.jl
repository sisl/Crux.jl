# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)

target(Q, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q(𝒟[:sp]), dims=1) # DQN
target(μ, Q, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(Q, 𝒟[:sp], action(μ, 𝒟[:sp])) # DDPG

q_predicted(Q, 𝒟) = sum(value(Q, 𝒟[:s]) .* 𝒟[:a], dims = 1)

td_loss(π, 𝒟, y, L, weighted = false) =  L(q_predicted(π, 𝒟), y, agg = weighted ? weighted_mean(𝒟[:weight]) : mean)

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)
