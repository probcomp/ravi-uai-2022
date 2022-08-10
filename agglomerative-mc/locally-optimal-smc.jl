using Random: shuffle

function locally_optimal_smc_step!(trace, i; retained=nothing)
    # Choose a cluster to reincorporate it into
    all_cluster_ids = collect(keys(trace.clusters))

    if isnothing(retained) || haskey(trace.clusters, retained.assignments[i])
        new_cluster_id = gensym()
    else
        new_cluster_id = retained.assignments[i]
    end
    new_cluster = singleton_cluster(trace, i)
    push!(all_cluster_ids, new_cluster_id)

    log_priors      = [crp_log_prior_predictive(trace, cluster_id, 1) for cluster_id in all_cluster_ids]
    log_likelihoods = [conditional_likelihood(trace, cluster_id, new_cluster) for cluster_id in all_cluster_ids]
    log_joints      = log_priors .+ log_likelihoods
    log_total       = logsumexp(log_joints)
    cluster_probabilities = exp.(log_joints .- log_total)
    if isnothing(retained)
        chosen_cluster_id = all_cluster_ids[rand(Categorical(cluster_probabilities))]
    else
        chosen_cluster_id = retained.assignments[i]
    end
    
    incorporate_point!(trace, i, chosen_cluster_id)
    
    # The weight is the logsumexp of the choices.
    # The reason is that the proposal is locally optimal:
    #   p(x) / q(x) = p(x) / (p(x)/sum(p(x) for x in xs)) = sum(p(x) for x in xs)
    return log_total
end


struct LocallyOptimalSMCOptions
    rejuvenation_frequency :: Int
    rejuvenation_iters :: Int
end

function run_smc(data::Vector, K::Int, hypers::Hyperparameters, C::Type, options=LocallyOptimalSMCOptions(20, 1); retained=nothing)

    K_unretained = isnothing(retained) ? K : K-1

    traces = [create_initial_dpmm_trace(hypers, C, data) for _ in 1:K]
    weights = zeros(K)

    for i in 1:length(data)

        # Advance
        for (j, trace) in enumerate(traces) 
            weights[j] += locally_optimal_smc_step!(trace, i; retained= j == K ? retained : nothing)

            # Move
            if options.rejuvenation_frequency > 0 && i % options.rejuvenation_frequency == 0
                if !isnothing(retained) && options.rejuvenation_iters > 0
                    @error "Code does not yet support Gibbs rejuvenation in cSMC"
                end
                for _ in 1:options.rejuvenation_iters
                   gibbs_sweep!(trace)
                end
            end
        end


        # Resample
        total_weight = logsumexp(weights)
        normalized_weights = exp.(weights .- total_weight)
        indices = [rand(Categorical(normalized_weights)) for _ in 1:K_unretained]
        !isnothing(retained) && push!(indices, K)
        traces = [copytrace(traces[i]) for i in indices]
        weights = fill(total_weight - log(K), K)
    end

    return traces, weights
end

function resample_unweighted_trace(traces, weights)
    normalized_weights = exp.(weights .- logsumexp(weights))
    return traces[rand(Categorical(normalized_weights))]
end