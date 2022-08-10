using StatsFuns: loggamma

struct GaussianHyperparameters <: Hyperparameters
    mu :: Float64
    kap :: Float64 
    alpha :: Float64
    beta :: Float64
end

mutable struct GaussianStatistics <: ClusterStats
    data_mean :: Float64
    data_sse  :: Float64
end

copystats(stats::GaussianStatistics) = GaussianStatistics(stats.data_mean, stats.data_sse)

const GaussianCluster = Cluster{GaussianStatistics, GaussianHyperparameters}

const GaussianTrace = DPMMTrace{GaussianCluster, GaussianHyperparameters}

# Initializing cluster stats

function empty_cluster_stats(trace::GaussianTrace)
    GaussianStatistics(0.0, 0.0)
end

# Updating sufficient statistics

function update_stats_after_adding_point!(cluster::GaussianCluster, x::Float64)
    N = length(cluster.members)
    x_minus_old_mean = x - cluster.stats.data_mean
    cluster.stats.data_mean += x_minus_old_mean / N
    cluster.stats.data_sse  += x_minus_old_mean * (x - cluster.stats.data_mean)
end

function update_stats_after_removing_point!(cluster::GaussianCluster, x::Float64)
    N = length(cluster.members)
    if N == 0
        cluster.stats.data_mean = 0.0
        cluster.stats.data_sse  = 0.0
    else
        x_minus_old_mean          =  x - cluster.stats.data_mean
        cluster.stats.data_mean  -= x_minus_old_mean / N
        x_minus_new_mean          =  x - cluster.stats.data_mean
        cluster.stats.data_sse    =  cluster.stats.data_sse - x_minus_new_mean * x_minus_old_mean
    end
end

function singleton_cluster(trace::GaussianTrace, index)
    GaussianCluster(trace, Set(index), GaussianStatistics(trace.data[index], 0.0))
end

# If cluster_id is new, gives the marginal likelihood.
function conditional_likelihood(trace::GaussianTrace, cluster_id::Symbol, provisional_cluster::GaussianCluster)
    hypers = trace.hypers

    existing_cluster = haskey(trace.clusters, cluster_id) ? trace.clusters[cluster_id] : Cluster(trace, Set{Int}(), empty_cluster_stats(trace))

    n = length(existing_cluster.members)
    if n == 0 # fast path
        kappa_n, alpha_n, beta_n = hypers.kap, hypers.alpha, hypers.beta
    else
        kappa_n = hypers.kap + n
        alpha_n = hypers.alpha + n/2
        beta_n  = hypers.beta + 0.5 * existing_cluster.stats.data_sse + (hypers.kap * n * (existing_cluster.stats.data_mean - hypers.mu)^2)/(2*kappa_n)
    end

    m = length(provisional_cluster.members)
    combined_mean = (n * existing_cluster.stats.data_mean + m * provisional_cluster.stats.data_mean) / (n + m)
    combined_sse  = provisional_cluster.stats.data_sse + existing_cluster.stats.data_sse + n * (existing_cluster.stats.data_mean - combined_mean)^2 + m * (provisional_cluster.stats.data_mean - combined_mean)^2

    tot = m + n
    kappa_tot = hypers.kap + tot
    alpha_tot = hypers.alpha + tot/2
    beta_tot = hypers.beta + 0.5 * combined_sse + (hypers.kap * tot * (combined_mean - hypers.mu)^2) / (2*kappa_tot)

    loggamma(alpha_tot) - loggamma(alpha_n) + log(beta_n) * alpha_n - log(beta_tot) * alpha_tot + 0.5 * (log(kappa_n) - log(kappa_tot)) - (m/2)*log(2pi)
end

# Update stats of existing cluster to include new datapoints
function merge_stats!(trace::GaussianTrace, existing_cluster_id::Symbol, new_cluster::GaussianCluster)
    cluster = trace.clusters[existing_cluster_id]
    n = length(cluster.members)
    m = length(new_cluster.members)

    combined_mean = (n * cluster.stats.data_mean + m * new_cluster.stats.data_mean) / (n + m)
    combined_sse  = new_cluster.stats.data_sse + cluster.stats.data_sse + n * (cluster.stats.data_mean - combined_mean)^2 + m * (new_cluster.stats.data_mean - combined_mean)^2
    cluster.stats.data_mean = combined_mean
    cluster.stats.data_sse = combined_sse
end


function summarize(trace::GaussianTrace)
    [(length(cluster.members), sort(collect(cluster.members)), 
     cluster.stats.data_mean, 
     sqrt(cluster.stats.data_sse / length(cluster.members))) 
     for (_, cluster) in trace.clusters]
end