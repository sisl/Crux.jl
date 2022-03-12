using CUDA, Crux

## Only run CUDA tests if it seems to be working properly
USE_CUDA = false
try 
	cu(zeros(Float32, 10, 10))
	USE_CUDA = true
	CUDA.allowscalar(false)
catch end

## Run basic functionality tests
include("spaces_tests.jl")
USE_CUDA && include("devices_tests.jl")
include("util_tests.jl")
include("experience_buffer_tests.jl")
include("sampler_tests.jl")
include("policy_tests.jl")
include("logging_tests.jl")

## Extras tests
include("extras_tests.jl")

## Solvers tests
include("solver_tests.jl")

