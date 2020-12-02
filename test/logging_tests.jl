using Crux
using Test

## elapsed
@test Crux.elapsed(1000, 100) 
@test !Crux.elapsed(1001, 100) 
@test Crux.elapsed(47001:47501, 500)
@test !Crux.elapsed(47001:47499, 500)


