using Shard
using Test

## elapsed
@test Shard.elapsed(1000, 100) 
@test !Shard.elapsed(1001, 100) 
@test Shard.elapsed(47001:47501, 500)
@test !Shard.elapsed(47001:47499, 500)


