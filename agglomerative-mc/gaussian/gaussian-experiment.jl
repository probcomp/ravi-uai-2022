using Distributions: Gamma, Normal
using Plots, StatsPlots
using Gen: logsumexp
using CSV, DataFrames
import StatsBase

include("../shared.jl")
include("../dpmm-data-structures.jl")
include("gaussian-clusters.jl")
include("../agglomerative-mc.jl")
include("../locally-optimal-smc.jl")
include("../gibbs.jl")
include("gaussian-visualization.jl")

# Hyperparameters
HYPERS = GaussianHyperparameters(0, 1 / 100, 1 / 2, 1 / 2)

# Create many synthetic datasets
function simulate_crp(alpha, n)
    tables = [1]
    for i in 2:n
        probs = [tables..., alpha] ./ (i - 1 + alpha)
        next_table = rand(Categorical(probs))
        if next_table > length(tables)
            push!(tables, 1)
        else
            tables[next_table] += 1
        end
    end
    return tables
end
function generate_synthetic_dataset(hypers, alpha, n)
    partition = simulate_crp(alpha, n)
    precs = [rand(Gamma(hypers.alpha, 1/hypers.beta)) for _ in partition]
    mus   = [rand(Normal(hypers.mu, sqrt(1/prec * 1/hypers.kap))) for prec in precs]
    data  = vcat([[rand(Normal(mu, sqrt(1/prec))) for i in 1:n] for (n, prec, mu) in zip(partition, precs, mus)]...)  
    shuffle(data)
end

function get_smc_result(dataset, K)
    logmeanexp(run_smc(dataset, K, HYPERS, GaussianCluster, LocallyOptimalSMCOptions(10, 0))[2])
end


# Synthetic data
using Serialization
# Load generated dataset
synthetic_dataset = deserialize("../../datasets/synthetic_data.log")
# ...or make a new one, by uncommenting the next line
# synthetic_dataset = generate_synthetic_dataset(HYPERS, 0.1, 100)
smc_results = [logmeanexp(run_smc(synthetic_dataset, 100, HYPERS, GaussianCluster, LocallyOptimalSMCOptions(10, 0))[2]) for _ in 1:30]
println("SMC results (synthetic): $(StatsBase.mean(smc_results)) +/- $(StatsBase.std(smc_results))")
amc_results = [logmeanexp([run_amc(HYPERS, GaussianCluster, synthetic_dataset; verbose=false, temperature=1.0, K=10).Z for _ in 1:3]) for _ in 1:30]
println("AMC results (synthetic): $(StatsBase.mean(amc_results)) +/- $(StatsBase.std(amc_results))")

# Galaxy data
galaxies_data = map(Float64, CSV.read("../../datasets/galaxies.csv", DataFrame)[!, 1])
smc_results = [logmeanexp(run_smc(galaxies_data, 100, HYPERS, GaussianCluster, LocallyOptimalSMCOptions(10, 0))[2]) for _ in 1:100]
println("SMC results (galaxies): $(StatsBase.mean(smc_results)) +/- $(StatsBase.std(smc_results))")
amc_results = [logmeanexp([run_amc(HYPERS, GaussianCluster, galaxies_data; verbose=false, temperature=1.0, K=10).Z for _ in 1:3]) for _ in 1:100]
println("AMC results (galaxies): $(StatsBase.mean(amc_results)) +/- $(StatsBase.std(amc_results))")
