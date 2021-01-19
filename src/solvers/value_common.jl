# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)

target(Q, 𝒟, γ::Float32) = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* maximum(Q(𝒟[:sp]), dims=1)

q_predicted(Q, 𝒟) = sum(value(Q, 𝒟[:s]) .* 𝒟[:a], dims = 1)

function td_loss(π, 𝒟, y, L, weighted = false; info = Dict())
    Q = q_predicted(π, 𝒟) 
    
    # Store useful information
    ignore() do
        info[:avg_Q] = mean(Q)
    end
    
    L(Q, y, agg = weighted ? weighted_mean(𝒟[:weight]) : mean)
end

td_error(π, 𝒟, y) = abs.(q_predicted(π, 𝒟) .- y)
