using CUDA, Crux
CUDA.allowscalar(false)
Crux.set_crux_warnings(false)
include("util_tests.jl")
include("experience_buffer_tests.jl")
include("sampler_tests.jl")
include("policy_tests.jl")
include("logging_tests.jl")

## Extras tests
include("extras_tests.jl")

## Solvers tests
include("solver_tests.jl")

