include("../src/includes.jl")
using Test

@test elapsed(1000, 100) 
@test elapsed(47501, 500, last_i = 47001)