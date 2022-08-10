using Colors

function visualize_trace(trace::DPMMTrace{GaussianCluster, GaussianHyperparameters}, title="Inferred clusters")
    println(length(filter(x -> length(x[2].members) > 1, trace.clusters))+1)
    colors = range(HSV(0,1,1), stop=HSV(-240,1,1), length=length(filter(x -> length(x[2].members) > 1, trace.clusters))+1)
    colormap = Dict(zip(sort(collect(keys(filter(x -> length(x[2].members) > 1, trace.clusters)))), colors))
    scatter(1:length(trace.data), trace.data, color=map(x -> length(trace.clusters[x].members) == 1 ? "black" : colormap[x], trace.assignments), markersize=10, title=title, label=nothing)
end