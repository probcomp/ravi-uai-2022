using Distributions: Categorical

function merge_clusters!(trace::DPMMTrace{Cluster{S,H}, H}, left_cluster_id, right_cluster_id) where {S,H}
    merge_stats!(trace, left_cluster_id, trace.clusters[right_cluster_id])
    push!(trace.clusters[left_cluster_id].members, trace.clusters[right_cluster_id].members...)
    for index in trace.clusters[right_cluster_id].members
        trace.assignments[index] = left_cluster_id
    end
    delete!(trace.clusters, right_cluster_id)    
end

function merge_is_inconsistent_with_final_trace(t, i, j)
    # All members of i and j must be part of the same cluster in the final trace.
    # However, we know, if we've gotten here, that both i and j are consistent.
    # So we actually only need to check one representative of each!
    t.assignments[first(i.members)] != t.assignments[first(j.members)]
end

function invalidate_cache!(cache, i, j, trace)
    for key in keys(trace.clusters)
        delete!(cache, (i, key))
        delete!(cache, (key, i))
        delete!(cache, (key, j))
        delete!(cache, (j, key))
    end
end

function compute_missing_cache_entries!(trace, cache; target_trace=nothing)
    for (i, cluster_i) in trace.clusters
        for (j, cluster_j) in trace.clusters
            if haskey(cache, (i,j)) || haskey(cache, (j,i)) || i == j
                continue
            end

            if !isnothing(target_trace) && merge_is_inconsistent_with_final_trace(target_trace, cluster_i, cluster_j)
                continue
            end

            cache[(i, j)] = 0.0

            # Compute delta to CRP prior: 
            N = length(trace.data) # for Agglomerative MCMC, makes sense to use trace.data here and not trace.active.
                                   # In other settings, e.g. as a rejuvenation move in the SMC algorithm, would only want to count trace.active.
            total = N - length(cluster_j.members)
            at_i = length(cluster_i.members)
            cache[(i, j)] -= (log(trace.alpha) - log(total + trace.alpha))
            for k in 1:length(cluster_j.members)
                cache[(i, j)] += log(at_i) - log(total + trace.alpha)
                if k > 1
                    cache[(i, j)] -= (log(k-1) - log(total + trace.alpha))
                end
                total += 1
                at_i  += 1 
            end

            # Compute delta to likelihood
            cache[(i, j)] -= conditional_likelihood(trace, gensym(), cluster_j)
            cache[(i, j)] += conditional_likelihood(trace, i, cluster_j)
        end
    end
end

function amc_step!(trace; target_trace=nothing, choice=nothing, cache=Dict{Tuple{Symbol, Symbol}, Float64}(), temperature=1.0)
    # If we have a target trace, check if we've already met it.
    # If so we can have no more merges.
    if !isnothing(target_trace)
        if isnothing(choice) && length(target_trace.clusters) == length(trace.clusters)
            return :done, 0.0
        elseif choice == :done
            return 0.0
        end
    end

    # If there is only one cluster, we're also done
    if length(trace.clusters) == 1
#        @warn "Got to length 1 check, with target_trace=$(!isnothing(target_trace))"
        if isnothing(choice)
            return :done, 0.0
        else
            return 0.0
        end
    end

    # Consider every pairwise merge
    compute_missing_cache_entries!(trace, cache; target_trace=target_trace)

    # If we are not aiming for a particular target, stopping is always an option;
    # if we are, we know we have to keep going.
    all_logprobs = isnothing(target_trace) ? [values(cache)..., 0.0] : collect(values(cache))
    all_logprobs .*= temperature 
    logZ = logsumexp(all_logprobs)
    
    if !isnothing(choice)
        if choice == :done
            return last(all_logprobs) - logZ
        end

        c1, c2 = choice
        merge_clusters!(trace, c1, c2)
        if !haskey(cache, choice)
            choice = (c2, c1)
        end

        score = temperature * cache[choice] - logZ
        invalidate_cache!(cache, c1, c2, trace)

        return score
    end

    all_logprobs .-= logZ

    i = rand(Categorical(exp.(all_logprobs)))
    if isnothing(target_trace) && i == length(all_logprobs)
        return :done, all_logprobs[i]
    else
        merge_keys  = collect(keys(cache))
        c1, c2 = merge_keys[i]
        merge_clusters!(trace, c1, c2)
        invalidate_cache!(cache, c1, c2, trace)
        return merge_keys[i], all_logprobs[i]
    end
end

struct RetainedParticle
    choices::Vector
    q_weights::Vector{Float64}
end

function effective_sample_size(log_normalized_weights::Vector{Float64})
    log_ess = -logsumexp(2. * log_normalized_weights)
    return exp(log_ess)
end

# Runs SMC or cSMC meta-inference
# When ess_threshold is 0, becomes SIR
# Returns an estimate of q(observed_trace)--either unbiased, or unbiased for the reciprocal (if cSMC)
function run_smc_meta(starting_trace, observed_trace; forward_cache=Dict{Tuple{Symbol,Symbol},Float64}(), K=1, temperature=1.0, retained=nothing, ess_threshold=K/4)
    compute_missing_cache_entries!(starting_trace, forward_cache; target_trace=observed_trace)

    # "SMC" meta-inference
    K_unretained = isnothing(retained) ? K : K-1
    h_caches = [copy(forward_cache) for k in 1:K]
    q_caches = [copy(forward_cache) for k in 1:K_unretained]
    h_traces = [copytrace(starting_trace) for k in 1:K]
    q_traces = [copytrace(starting_trace) for k in 1:K_unretained]

    weights  = zeros(K)

    total_merges_necessary =  length(starting_trace.clusters) - length(observed_trace.clusters)
    for _ in 1:(total_merges_necessary + 1)
        # Advance all
        for k in 1:K
            if !isnothing(retained) && k==K
                choice = popfirst!(retained.choices)
                dh = amc_step!(h_traces[k]; cache=h_caches[k], temperature=temperature, target_trace=observed_trace, choice=choice)
                dq = popfirst!(retained.q_weights)
            else
                choice, dh = amc_step!(h_traces[k]; cache=h_caches[k], temperature=temperature, target_trace=observed_trace)
                dq = amc_step!(q_traces[k]; cache=q_caches[k], temperature=temperature, choice=choice)
                # println(dh, dq, dh-dq)
            end
            weights[k] += dq - dh
            #println(weights[k])
        end
        # Resample the first K_unretained particles
        logZ = logsumexp(weights)
        # println(weights)
        log_normalized_weights = weights .- logZ
        if effective_sample_size(log_normalized_weights) < ess_threshold
            # println("Resampling! $(effective_sample_size(log_normalized_weights)) < $(ess_threshold)")
            ancestor_indices = [rand(Categorical(exp.(weights .- logZ))) for i in 1:K_unretained]
            q_ancestor_indices = copy(ancestor_indices)
            if !isnothing(retained)
                push!(ancestor_indices, K)
                push!(q_caches, copy(last(h_caches)))
                push!(q_traces, copytrace(last(h_traces)))
            end
            
            h_caches, q_caches, h_traces, q_traces = map(copy, h_caches[ancestor_indices]), map(copy, q_caches[q_ancestor_indices]), map(copytrace, h_traces[ancestor_indices]), map(copytrace, q_traces[q_ancestor_indices])

            weights = ones(K) * (logZ - log(K))
        end
    end
    return logmeanexp(weights)
end

# Set ESS threshold to 0.0 for SIR meta-inference, instead of SMC meta-inference
function run_amc(hypers::H, cluster_type::Type{C}, data; verbose=true, temperature=1.0, K=1, ess_threshold=K/4, starting_trace=nothing, return_initial_cache=false) where {C,H}
    if isnothing(starting_trace)
        t = create_singletons_dpmm_trace(hypers, cluster_type, data)
    else
        t = starting_trace
    end
    starting_trace = copytrace(t)

    cache = Dict{Tuple{Symbol, Symbol}, Float64}()
    saved_cache = Dict{Tuple{Symbol, Symbol}, Float64}()
    # Inference
    L = length(t.clusters) + 1
    q_weights = []
    history = []
    first_step = true
    while L > length(t.clusters)
        # if verbose
        #     println("$L clusters")
        # end
        L = length(t.clusters)
        choice, dq = amc_step!(t; cache=cache, temperature=temperature)
        push!(q_weights, dq)
        push!(history, choice)
        if first_step
            saved_cache = copy(cache)
            first_step = false
        end
    end

    # "SMC" meta-inference
    q = run_smc_meta(starting_trace, t; forward_cache=saved_cache, 
                        retained=RetainedParticle(history, q_weights), K=K, temperature=temperature, ess_threshold=ess_threshold)
    p = log_joint(t)
    if return_initial_cache
        return (trace=t, p=p, Z=p-q, cache=saved_cache)
    else
        return (trace=t, p=p, Z=p-q)
    end
end
