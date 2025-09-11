using Test, CUDA, Crux

## Only run CUDA tests if it seems to be working properly
USE_CUDA = false
try
	cu(zeros(Float32, 10, 10))
	USE_CUDA = true
	CUDA.allowscalar(false)
catch end

## Run functionality tests
@testset "spaces" begin
	include("spaces_tests.jl")
end
USE_CUDA && @testset "devices" begin
	include("devices_tests.jl")
end
@testset "util" begin
	include("util_tests.jl")
end
@testset "buffer" begin
	include("experience_buffer_tests.jl")
end
@testset "policy" begin
	include("policy_tests.jl")
end
@testset "logging" begin
	include("logging_tests.jl")
end
@testset "extras" begin
	include("extras_tests.jl")
end
@testset "solver" begin
	include("gym/solver_tests.jl")
end
@testset "sampler" begin
	include("gym/sampler_tests.jl")
end
