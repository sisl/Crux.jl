using Flux, Statistics, MLDatasets, Random
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using CUDA

function mlp_model(imgsize = (28,28,1), nclasses = 10)
    return Chain(Dense(prod(imgsize), 400, relu), Dense(400,400, relu), Dense(400, nclasses)) |> gpu
end

function conv_model(imgsize = (28,28,1), nclasses = 10)
    cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))	

    return Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(cnn_output_size), 10)) |> gpu
end

augment(x) = x .+ gpu(0.1f0*randn(Float32, size(x)))

compare(y, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(model, data) = mean(compare(data[2], cpu(model)(data[1])))

function gen_data(;permute = true, 
                   seed = 0, 
                   er_with_seeds = nothing,
                   ρ_er = 0.5,
                   flatten = true, 
                   train_on_gpu = true, 
                   test_on_gpu = false, 
                   imgsize = (28,28,1), 
                   nclasses = 10, 
                   batchsize = 128)
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y  = MNIST.testdata(Float32)
    Ntrain = length(train_y)
    
    N = prod(imgsize)
    permutation = (permute) ? randperm(MersenneTwister(seed), N) : 1:N

    
    train_y = Float32.(Flux.onehotbatch(train_y, 0:nclasses-1))
    train_x = reshape(train_x, N, :)
    
    if !isnothing(er_with_seeds)
        train_x_p = train_x[permutation, :]
        perms = [randperm(MersenneTwister(seed), N) for seed in er_with_seeds]
        for i=1:Ntrain
            if rand() < ρ_er
                train_x_p[:, i] = train_x[rand(perms), i]
            end
        end
        train_x = train_x_p
    else
        train_x = train_x[permutation, :]
    end
    
    test_x = reshape(test_x, N, :)[permutation, :]
    test_y = Float32.(Flux.onehotbatch(test_y, 0:nclasses-1)) 
    
    if !flatten
        train_x = reshape(train_x, imgsize..., :)
        test_x = reshape(test_x, imgsize..., :)
    end 
    if train_on_gpu
        train_x = train_x |> gpu
        train_y = train_y |> gpu
    end
    if test_on_gpu
        test_x = test_x |> gpu
        test_y = test_y |> gpu
    end
    DataLoader((train_x, train_y), batchsize = batchsize, shuffle = true), (test_x, test_y)
end

function train_classifier!(model, data, opt;)	
    Flux.train!((x,y) -> logitcrossentropy(model(augment(x)), y), params(model), data, opt)
end

## Testcase for stochastic gradient descent
train_data1, test_data1 = gen_data(seed = 0)
train_data2, test_data2 = gen_data(seed = 1)
train_data3, test_data3 = gen_data(seed = 2)
mlp = mlp_model()

train_data = [train_data1, train_data2, train_data3]
test_data = [test_data1, test_data2, test_data3]

results = continual_learning!(mlp, train_classifier!, train_data, test_data, accuracy)

p = plot_results(results, title = "Scrambed MNIST Classification Tasks")
savefig("SGD_on_MNIST_Scrambled.pdf")

## Testcase for Experience replay
train_data1, test_data1 = gen_data(seed = 0)
train_data2, test_data2 = gen_data(seed = 1, er_with_seeds = [0])
train_data3, test_data3 = gen_data(seed = 2, er_with_seeds = [0, 1])
mlp = mlp_model()

train_data = [train_data1, train_data2, train_data3]
test_data = [test_data1, test_data2, test_data3]

results = continual_learning!(mlp, train_classifier!, train_data, test_data, accuracy)

p = plot_results(results, title = "Scrambed MNIST Classification Tasks w/ ER")
savefig("ER_on_MNIST_Scrambled.pdf")


