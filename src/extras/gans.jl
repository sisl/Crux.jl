abstract type GANLoss end

# Loss that includes the generator calls out to loss that just takes in generated values
Lᴰ(t::GANLoss, G, D, z, x; wD = 1f0, yG = (), yD = ()) = Lᴰ(t, D, x, G(z, yG...), wD = wD, yG = yG, yD = yD)

## Binary Cross Entropy Loss
struct GAN_BCELoss <: GANLoss end
const LBCE = Flux.Losses.logitbinarycrossentropy
Lᴰ(t::GAN_BCELoss, D, x, xtilde; wD = 1f0, yG = (), yD = ()) = LBCE(D(x, yD...), wD) + LBCE(D(xtilde, yG...), 0f0)
Lᴳ(t::GAN_BCELoss, G, D, z; yG = ()) = LBCE(D(G(z, yG...), yG...), 1f0)

## Least Squares Loss
struct GAN_LSLoss <: GANLoss end
Lᴰ(t::GAN_LSLoss, D, x, xtilde; wD = 1f0, yG = (), yD = ()) = Flux.mse(D(x, yD...), wD) + Flux.mse(D(xtilde, yG...), 0f0)
Lᴳ(t::GAN_LSLoss, G, D, z; yG = ()) = Flux.mse(D(G(z, yG...), yG...), 1f0)

## Hinge Loss
struct GAN_HingeLoss <: GANLoss end
Lᴰ(t::GAN_HingeLoss, D, x, xtilde; wD = 1f0, yG = (), yD = ()) = mean(relu.(1f0 .+ (-2f0 .* wD .+ 1f0) .* D(x, yD...))) + mean(relu.(1f0 .+ D(xtilde, yG...)))
Lᴳ(t::GAN_HingeLoss, G, D, z; yG = ()) = -mean(D(G(z, yG...), yG...)) 

## WLoss
struct GAN_WLoss <: GANLoss end
Lᴰ(t::GAN_WLoss, D, x, xtilde; wD = 1f0, yG = (), yD = ()) = mean(D(xtilde, yG...)) -  mean((wD .* 2f0 .- 1f0) .* D(x, yD...))
Lᴳ(t::GAN_WLoss, G, D, z; yG = ()) = -mean(D(G(z, yG...), yG...)) 


## WGAN-GP
@with_kw struct GAN_WLossGP <: GANLoss
	λ::Float32 = 10f0 # gp parameter
	noise_range::Float32 = 1f0 # Range of the ϵ parameter that picks a point to evalutate the gradient (smaller is closer to "real" data)
end

function gradient_penalty(D, x, y)
	B = size(x, ndims(x))
	l, b = Flux.pullback(() -> D(x, y), Flux.params(x, y))
	grads = b(ones(Float32, size(l)) |> device(x))
	mean((sqrt.(sum(reshape(grads[x], :, B).^2, dims = 1) .+ sum(grads[y].^2, dims = 1)) .- 1f0).^2)
end

function gradient_penalty(D, x)
	B = size(x, ndims(x))
	l, b = Flux.pullback(() -> D(x), Flux.params(x))
	grads = b(ones(Float32, size(l)) |> device(x))
	Flux.mean((sqrt.(sum(reshape(grads[x], :, B).^2, dims = 1) .+ sum(grads[y].^2, dims = 1)) .- 1f0).^2)
end

function Lᴰ(t::GAN_WLossGP, D, x, xtilde; wD = 1f0, yG = (), yD = ())
	loss = mean(D(xtilde, yG...)) -  mean((wD .* 2f0 .- 1f0) .* D(x, yD...))
	
	# Compute gradient penalty
	B = size(x, ndims(x))
	ϵ = Zygote.ignore(() -> Float32.(rand(Uniform(0f0, t.noise_range), 1, B)) |> device(x))
	ϵx = reshape(ϵ, ones(Int,  ndims(x)-1)..., B)
	xhat = ϵx .* xtilde .+ (1f0 .- ϵx) .* x
	loss += (length(yG) == 1) ? t.λ*gradient_penalty(D, xhat, ϵ .* yG[1] .+ (1f0 .- ϵ) .* yD[1]) : t.λ*gradient_penalty(D, xhat)
	loss
end

Lᴳ(t::GAN_WLossGP, G, D, z; yG = ()) = -mean(D(G(z, yG...), yG...)) 


