using Distributions, StatsBase

function cross_entropy(f, P; k=1, m=50, m_extra=20, m_elite=m, base_dist=Uniform(-1,1))
    for _=1:k
        # Sample from the current distribution
        samples = clamp.(rand(P, m), -1f0, 1f0)
        dim = size(samples, 1)
        
        # Add some noise to avoid the distribution collapsing entirely
        samples .+= rand(Uniform(-1f-3, 1f-3), size(samples)...)
        
        # Add some extra samples from an exploration policy to account for dramatic shifts from previous iterations
        samples = hcat(samples, rand(base_dist, dim, m_extra))
        
        vals = [f(samples[:,i]) for i=1:size(samples,2)]
        weights = softmax(-vals)
        
        
        order = sortperm(vals)
        elite_samples = Float64.(samples[:, order[1:m_elite]])
        try
            P = fit(MvNormal, elite_samples, weights[order[1:m_elite]])
        catch
            println("weights: ", weights)
            @error "Error in fitting MVNormal distribution"
        end
    end
    P
end

function mcmc(f, P; Q=(sz...)->0.01*randn(sz...), k=1, m=size(P,2), m_extra=20, base_dist = Uniform(-1,1))
    dim, Npart=size(P)
    @assert Npart == m
    for _=1:k
        # Perturb particles and add some additional ones
        # P = Float32.(clamp.(P .+ Q(dim, Npart), -1, 1))
        P = hcat(P, rand(base_dist, dim, m_extra))

        # compute the particle values
        vals = [f(P[:,i]) for i=1:size(P,2)]
        weights = exp.(-vals)
        
        # resample
        samples = sample(1:Npart+m_extra, Weights(weights), Npart)
        P = P[:, samples]
    end
    P
end

best_estimate(d::MvNormal) = d.μ
uncertainty(d::MvNormal) = det(d.Σ)
best_estimate(P::Array) = mean(P, dims=2)
uncertainty(P::Array) = mean(std(P, dims=2))

# # Demo the inference
# target_dist = MvNormal([-0.5,-0.5], [0.1, 0.1])
# foo(z) = -logpdf(target_dist, z)
# x = -2:0.01:2
# y = x
# 
# ## Particle filter
# 
# contour(x, y, (x,y)->exp(logpdf(target_dist, [x,y])))
# P = randn(2,100) # Initial particle set
# scatter!(P[1,:], P[2,:], label="iteration 0")
# P = mcmc(foo, P)
# scatter!(P[1,:], P[2,:], label="iteration 1")
# P = mcmc(foo, P)
# scatter!(P[1,:], P[2,:], label="iteration 2")
# 
# println("best estimate: ", best_estimate(P))
# 
# ## cross entropy
# p1 = contour(x, y, (x,y)->exp(logpdf(target_dist, [x,y])), cbar=false)
# P = MvNormal([0,0], Diagonal([1.,1.]))
# p2 = contour(x, y, (x,y)->exp(logpdf(P, [x,y])), seriescolor=:blues)
# P = cross_entropy(foo, P)
# p3 = contour(x, y, (x,y)->exp(logpdf(P, [x,y])), seriescolor=:blues)
# P = cross_entropy(foo, P)
# p4 = contour(x, y, (x,y)->exp(logpdf(P, [x,y])), seriescolor=:blues)
# 
# plot(p1, p2, p3, p4)
# 
# println("best estimate: ", best_estimate(P))
