using Shard
using Test

## elapsed
@test elapsed(1000, 100) 
@test !elapsed(1001, 100) 
@test elapsed(47001:47501, 500)
@test !elapsed(47001:47499, 500)


