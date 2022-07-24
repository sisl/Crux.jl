function gradient_penalty(D, x; target::Float32=1f0)
	B = size(x, ndims(x))
	l, b = Flux.pullback(D, x)
	ones_vec = ignore_derivatives() do 
					ones(Float32, size(l)) |> device(x) 
				end
	grads = b(ones_vec)
    Flux.mean((sqrt.(sum(reshape(grads[1], :, B).^2, dims = 1)) .- target).^2)
end

function gradient_penalty(D, x, xtilde; target::Float32=1f0)
	B = size(x, ndims(x))
	ϵ = ignore_derivatives(() -> Float32.(rand(Uniform(), 1, B)) |> device(x))
	ϵx = reshape(ϵ, ones(Int,  ndims(x)-1)..., B)
	xhat = ϵx .* xtilde .+ (1f0 .- ϵx) .* x
	gradient_penalty(D, xhat, target=target)
end
	

