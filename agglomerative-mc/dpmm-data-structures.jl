abstract type ClusterStats end
abstract type Hyperparameters end

# Tracks the state of a particle in the DPMM model.
# The `data` field references a common data vector shared
# across all particles, but each particle may represent
# a clustering of only some of the data, as indicated
# by the `active` set of indices.
struct DPMMTrace{C,H<:Hyperparameters}
    data     :: Vector
    active   :: Set{Int}
    clusters :: Dict{Symbol, C}
    assignments :: Vector{Symbol}

    # All hyperparameters
    hypers :: H
    alpha :: Float64
end

function create_initial_dpmm_trace(hypers::H, cluster_type::Type{C}, data) where {C, H}
    DPMMTrace{C, H}(data, Set(), Dict{Symbol, C}(), Vector{Symbol}(undef, length(data)), hypers, 1.0)
end

# Represents a single cluster within a particle for the
# DPMM model.
struct Cluster{T<:ClusterStats, H<:Hyperparameters}
    trace   :: DPMMTrace{Cluster{T,H}}
    members :: Set{Int}
    stats   :: T
end


function incorporate_cluster!(trace::DPMMTrace{Cluster{S, H}, H}, existing_cluster_id, new_cluster) where {S, H}
    if !haskey(trace.clusters, existing_cluster_id)
        trace.clusters[existing_cluster_id] = Cluster{S,H}(trace, copy(new_cluster.members), copystats(new_cluster.stats))
    else
        merge_stats!(trace, existing_cluster_id, new_cluster)
        push!(trace.clusters[existing_cluster_id].members, new_cluster.members...)
    end

    push!(trace.active, new_cluster.members...)
    for index in new_cluster.members
        trace.assignments[index] = existing_cluster_id
    end
end


function create_singletons_dpmm_trace(hypers::H, cluster_type::Type{C}, data) where {C, H}
    t = create_initial_dpmm_trace(hypers, cluster_type, data)

    for i in 1:length(data)
        # Add singleton cluster that x is in
        cluster_name = gensym()
        incorporate_cluster!(t, cluster_name, singleton_cluster(t, i))
    end

    return t
end


# Unincorporate a single point from a trace.
function unincorporate_point!(trace::DPMMTrace, index::Int)
    # Set datapoint to inactive
    delete!(trace.active, index)

    # Find datapoint's current cluster
    current_cluster_id = trace.assignments[index]
    current_cluster    = trace.clusters[current_cluster_id]

    # If it was the last point in its cluster, delete the cluster
    if length(current_cluster.members) == 1
        delete!(trace.clusters, current_cluster_id)
        return
    end

    # Else, update cluster not to contain the datapoint
    delete!(current_cluster.members, index)
    update_stats_after_removing_point!(current_cluster, trace.data[index])
end

# Incorporate a single point into a trace, with a certain cluster
function incorporate_point!(trace::DPMMTrace, index::Int, cluster_id::Symbol)
    push!(trace.active, index)

    if !haskey(trace.clusters, cluster_id)
        trace.clusters[cluster_id] = Cluster(trace, Set{Int}(), empty_cluster_stats(trace))
    end
    
    cluster = trace.clusters[cluster_id]
    push!(cluster.members, index)
    trace.assignments[index] = cluster_id
    
    update_stats_after_adding_point!(cluster, trace.data[index])
end

function create_giant_cluster_dpmm_trace(hypers::H, cluster_type::Type{C}, data) where {C,H}
    t = create_initial_dpmm_trace(hypers, cluster_type, data)
    cluster_name = gensym()
    for i in 1:length(data)
        incorporate_point!(t, i, cluster_name)
    end
    return t
end

function copytrace(trace::DPMMTrace{Cluster{S, H},H}) where {S, H}
    new_trace = DPMMTrace{Cluster{S, H}, H}(trace.data, copy(trace.active), Dict{Symbol, Cluster}(), copy(trace.assignments), trace.hypers, trace.alpha)
    for (id, cluster) in trace.clusters
        new_trace.clusters[id] = Cluster(new_trace, copy(cluster.members), copystats(cluster.stats))
    end
    return new_trace
end



# Conditional prior probability of N new customers sitting at cluster
# cluster_id. If cluster_id is not a key of trace.clusters, return the
# probability of starting a new cluster, and N-1 subsequent customers 
# sitting there.
function crp_log_prior_predictive(trace, cluster_id, N)
    current_total_size = length(trace.active)
    alpha = trace.alpha

    # New cluster
    if !haskey(trace.clusters, cluster_id)
        new_cluster_prob = log(alpha) - log(alpha + current_total_size)
        remaining_customers_prob = N < 2 ? 0.0 : sum(log(i-1) - log(alpha + current_total_size + i - 1) for i in 2:N)
        return new_cluster_prob + remaining_customers_prob
    end
    
    # Existing cluster
    current_cluster_size = length(trace.clusters[cluster_id].members)
    return N < 1 ? 0.0 : sum(log(current_cluster_size + i - 1) - log(alpha + current_total_size + i - 1) for i in 1:N)
end

# Joint probability of all assignments under the CRP prior
function crp_log_joint(trace)
    lp = 0.0
    tot = 0
    alpha = trace.alpha

    for (_, cluster) in trace.clusters 
        lp += log(alpha) - log(tot + alpha)
        tot += 1
        for j in 2:length(cluster.members)
            lp += log(j-1) - log(tot + alpha)
            tot += 1
        end
    end

    return lp
end

# Joint probability of an entire trace
function log_joint(trace)
    # Initialize to prior CRP probability
    log_prob = crp_log_joint(trace)
    # Include data likelihood for each table
    for (_, cluster) in trace.clusters
        log_prob += conditional_likelihood(trace, gensym(), cluster)
    end
    return log_prob
end