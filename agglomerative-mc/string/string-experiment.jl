using CSV
using DataFrames: DataFrame

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("string-clusters.jl")
include("../locally-optimal-smc.jl")
include("../gibbs.jl")
include("../agglomerative-mc.jl")

dataset = "hospital"
dirty_table = CSV.File("../../datasets/$dataset.csv") |> DataFrame;
clean_table = CSV.File("../../datasets/$(dataset)_clean.csv") |> DataFrame;

possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                        for col in names(dirty_table))

string_dataset = dirty_table.MeasureName

HYPERS = make_hypers(unique(possibilities["MeasureName"]))

# Test vanilla SMC
vanilla_smc_results = []
for _ in 1:30
    @time traces, weights = run_smc([s for s in string_dataset], 32, HYPERS, StringCluster, LocallyOptimalSMCOptions(100, 1));
    resampled_trace = resample_unweighted_trace(traces, weights)
    push!(vanilla_smc_results, (weight=logmeanexp(weights), n_clusters=length(resampled_trace.clusters)))
end


# Test agglomerative MC
agg_mc_results = []
for _ in 1:30
    @time r = run_amc(HYPERS, StringCluster, string_dataset);
    push!(agg_mc_results, r)
end

println("Baseline: $(mean(map(x -> x.weight, vanilla_smc_results))) +/- $(StatsBase.std(map(x -> x.weight, vanilla_smc_results)))")
println("Agglom. MC: $(mean(map(x -> x.weight, agg_mc_results))) +/- $(StatsBase.std(map(x -> x.weight, agg_mc_results)))")