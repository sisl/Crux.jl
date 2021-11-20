using Flux
struct SirenDense
    weight
    bias
    ω0
end

Flux.@functor SirenDense

function SirenDense(din, dout; c=6, ω0=1, isfirst=false)
    w_std = isfirst ? (1 / din) : (sqrt(c / din) / ω0)
    dist = Uniform(-w_std, w_std)
    weight = Float32.(rand(dist, dout, din))
    bias = Float32.(rand(dist, dout))
    SirenDense(weight, bias, ω0)
end

function (a::SirenDense)(x)
  sin.(a.ω0 * (a.weight*x .+ a.bias))
end


struct ModulatedSiren
    sirens
    modulator
end

Flux.trainable(π::ModulatedSiren) = (Flux.trainable(π.sirens)..., Flux.trainable(π.modulator)...)

function (a::ModulatedSiren)(x, z)
    slayers = layers(a.sirens)
    mlayers = layers(a.modulator)
    @assert length(slayers) == 4
    @assert length(slayers) == length(mlayers) + 1

    h = z
    
    h = mlayers[1](h)
    x = slayers[1](x) .* h
    h = vcat(h, z)
    
    h = mlayers[2](h)
    x = slayers[2](x) .* h
    h = vcat(h, z)
    
    h = mlayers[3](h)
    x = slayers[3](x) .* h
    # h = vcat(h, z)
    # 
    # h = mlayers[4](h)
    # x = slayers[4](x) .* h
    # h = vcat(h, z)
    # 
    # h = mlayers[5](h)
    # x = slayers[5](x) .* h
    
    
    slayers[end](x)
end


