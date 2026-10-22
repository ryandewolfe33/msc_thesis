using Pkg
using ABCDGraphGenerator
using Random
# ABCDGraphGenerator can be installed from 
# https://github.com/bkamins/ABCDGraphGenerator.jl/blob/master/utils/abcd_oo_sampler.jl

n = 10000
nout = 1000
d = 8
ρ = 0.25

τ₁ = 2.5
d_min = 3
d_max = 1000
d_max_iter = 100

τ₂ = 1.5
c_min = 25
c_max = 1000
c_max_iter = 100

xis = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
etas = [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.]

runs = 10

# in what follows n is number of non-outlier nodes
n = n - nout

for ξ in xis
    for η in etas
        for run in 1:runs
            @info "ξ=$(ξ), η=$(η), #$(run)"
            degs = ABCDGraphGenerator.sample_degrees_oo(τ₁, d_min, d_max, n + nout, d_max_iter)
            coms = ABCDGraphGenerator.sample_communities_oo(τ₂, ceil(Int, c_min / η), floor(Int, c_max / η), n, c_max_iter)
            @assert sum(coms) == n
            pushfirst!(coms, nout)

            p = ABCDGraphGenerator.ABCDParamsOO(degs, coms, ξ, η, d, ρ)
            edges, clusters = ABCDGraphGenerator.gen_graph_oo(p)
            open("synthetic/graph_$(ξ)_$(η)_$(run).dat", "w") do io
                for (a, b) in sort!(collect(edges))
                    println(io, a, "\t", b)
                end
            end
            open("synthetic/labels_$(ξ)_$(η)_$(run).dat", "w") do io
                for (i, c) in enumerate(clusters)
                    println(io, i, "\t", c)
                end
            end
        end
    end
end