include("../src/includes.jl")
using Test


m = MultitaskDecaySchedule(10, [1,2,3])
l = LinearDecaySchedule(start = 1.0, stop = 0.1, steps = 10)

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-20)
end

m = MultitaskDecaySchedule(10, [1,2,1])

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-10)
end

@test m(31) == 0.1
@test m(0) == 1

