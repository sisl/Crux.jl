# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)

function td_loss(π, 𝒟, y, L, weighted = false; info = Dict())
    Q = value(π, 𝒟[:s], 𝒟[:a]) 
    
    # Store useful information
    ignore() do
        info[:avg_Q] = mean(Q)
    end
    
    L(Q, y, agg = weighted ? weighted_mean(𝒟[:weight]) : mean)
end

td_error(π, 𝒟, y) = abs.(value(π, 𝒟[:s], 𝒟[:a])  .- y)
