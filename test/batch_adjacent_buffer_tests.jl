using Crux
using Random
using Test


b = ExperienceBuffer(ContinuousSpace(2), ContinuousSpace(1), 100)
d = Dict(:s => 2*ones(2,50), :a => ones(Bool, 1,50), :sp => ones(2,50), :r => ones(1,50), :done => zeros(1,50), :weight=>zeros(1,50))
push!(b, d)