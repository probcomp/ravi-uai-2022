
# Perform Sequential Monte Carlo
struct RetainedParticle
    locations::Vector{Float64} # x_{M:-1:1} -- does not include final location x_{M+1}
    weights::Vector{Float64}   # does not include initial weight, just likelihoods M+1:-1:2
end

struct SMCParams
    K :: Int                    # number of particles, at least 1
    M :: Int                    # number of MCMC transitions
    ess_threshold :: Float64    # as a fraction of K. Set to 0 for SIR (no resampling)

    # takes in a step number, outputs a marginal... used to construct intermediate targets
    # we trust that the step marginal for step 1 is correct (the initial MCMC distribution).
    # there should be an estimate for M+1
    step_marginal_estimates :: Kernel
    target_marginals :: Bool
end

function SMCParams(K,M,ess,marginals)
    SMCParams(K, M, ess, marginals, false)
end

struct ConstantStepMarginalEstimator <: Kernel
    marginal :: ProbModule
end
(c::ConstantStepMarginalEstimator)(step) = c.marginal

constant_step_marginal_estimator(mcvi) = ConstantStepMarginalEstimator(mcvi.initial_proposal)

function smc_step!(particle, which_step, mcvi, step_marginals; retained=nothing, target_marginals=false)
    is_retained = !isnothing(retained)
    is_trajectory = particle isa Vector
    current = is_trajectory ? last(particle) : particle

    h = predict_previous_step(mcvi.metainference, current, which_step)

    if is_retained
        location, q_weight = popfirst!(retained.locations), popfirst!(retained.weights)
        h_weight = run(h; val=location)
    else
        location, h_weight = run(h)

        # Evaluate q's weight.
        target = target_marginals ? step_marginals(which_step+1) : mcvi.model
        q_weight = run(mcvi.mcmc_kernel(target, location); val=current)
    end

    model_correction = run(step_marginals(which_step);   val=location) - 
                       run(step_marginals(which_step+1); val=current)

    weight_increment = model_correction + q_weight - h_weight
    return is_trajectory ? (push!(particle, location), weight_increment) : (location, weight_increment)
end


function smc(final_value, 
    mcvi::MCVIAlgorithm, 
    options::SMCParams; 
    retained=nothing, 
    return_particles=false,
    return_weight_histories=false)


    particles = [return_particles ? [final_value] : final_value for _ in 1:options.K]
    weights   = fill(run(options.step_marginal_estimates(options.M+1); val=final_value), options.K)
    weight_histories = [[weight] for weight in weights]

    for i in options.M:-1:1
        # Advance each particle
        for j in 1:options.K
            particles[j], weight_increment = smc_step!(particles[j], i, mcvi, options.step_marginal_estimates;
                retained = (j == options.K) ? retained : nothing, target_marginals=options.target_marginals)
            weights[j] += weight_increment
            return_weight_histories && push!(weight_histories[j], weights[j])
        end

        # Maybe resample
        log_normalized_weights = weights .- logsumexp(weights)
        ess = Gen.effective_sample_size(log_normalized_weights)
        if ess < options.ess_threshold * options.K
            K_unretained     = isnothing(retained) ? options.K : options.K-1
            ancestor_indices = [Gen.random(categorical, exp.(log_normalized_weights)) for i in 1:K_unretained]
            !isnothing(retained) && push!(ancestor_indices, options.K)

            particles = map(copy, particles[ancestor_indices])
            weights   = ones(options.K) * logmeanexp(weights)
        end
    end

    if return_weight_histories
        return particles, weight_histories
    elseif return_particles
        return particles, weights .- logsumexp(weights), logmeanexp(weights)
    else
        return logmeanexp(weights)
    end
end

# Generate via MCMC, and return the point & a weight
function mcmc_propose(mcvi, smc_options; return_weight_histories=false, return_particles=false)
    history, weights = [], []
    
    # Initial location
    location, = run(mcvi.initial_proposal)
    push!(history, location)

    # Run MCMC
    for which_step = 2:smc_options.M+1
        location, weight = run(mcvi.mcmc_kernel(mcvi.model, location))
        push!(history, location)
        push!(weights, weight)
    end

    # Run SMC
    final_location = pop!(history)
    retained = RetainedParticle(reverse(history), reverse(weights))
    final_location, smc(final_location, mcvi, smc_options; retained=retained, return_particles=return_particles, return_weight_histories=return_weight_histories)
end