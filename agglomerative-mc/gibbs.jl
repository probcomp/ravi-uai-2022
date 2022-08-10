using Distributions: Categorical

function gibbs_sweep!(trace::DPMMTrace)
    # For each point in each cluster, consider reassigning
    active_points = collect(trace.active)

    for i in active_points
        # Remove the point from the dataset
        unincorporate_point!(trace, i)

        # Perform a 'locally optimal SMC step'
        locally_optimal_smc_step!(trace, i)
    end
end

function run_gibbs(T, hypers, C, dataset)
    trace = create_initial_dpmm_trace(hypers, C, dataset)

    # Incorporate each datapoint into its own cluster
    for i in 1:length(dataset)
        incorporate_point!(trace, i, gensym())
    end

    # Run Gibbs sweeps
    for _ in 1:T
        gibbs_sweep!(trace)
    end

    # Return final result
    return trace
end


function run_gibbs_2(T, hypers, C, data)
    trace = create_giant_cluster_dpmm_trace(hypers, C, data)

    # Run Gibbs sweeps
    for _ in 1:T
        gibbs_sweep!(trace)
    end

    # Return final result
    return trace
end