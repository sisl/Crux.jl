using Pkg
using Test

try
	using POMDPGym
catch e
	Pkg.add(url="https://github.com/ancorso/POMDPGym.jl")
	try
		using POMDPGym
	catch e
		@info "Error loading POMDPGym"
		throw(e)
	end
end
using CUDA, Crux

## Only run CUDA tests if it seems to be working properly
USE_CUDA = false
try
	cu(zeros(Float32, 10, 10))
	USE_CUDA = true
	CUDA.allowscalar(false)
catch e
	@info "Error loading CUDA, not performing GPU tests" e
end

## Run basic functionality tests
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
@testset "sampler" begin
	include("sampler_tests.jl")
end
@testset "policy" begin
	include("policy_tests.jl")
end
@testset "logging" begin
	include("logging_tests.jl")
end


## Extras tests
@testset "extras" begin
	include("extras_tests.jl")
end


## Solvers tests
@testset "solver" begin
	include("solver_tests.jl")
end
