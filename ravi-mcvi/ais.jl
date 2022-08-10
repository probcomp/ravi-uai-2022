struct AnnealedModel <: ProbModule
    target :: ProbModule
    prior  :: ProbModule
    beta   :: Float64
end

function run(annealed_model::AnnealedModel; val=nothing)
    if isnothing(val)
        if annealed_model.beta == 1
            return run(annealed_model.target)
        elseif annealed_model.beta == 0
            return run(annealed_model.prior)
        else
            @error "Cannot simulate from annealed model with beta=$(annealed_model.beta)"
        end
    end

    # Note that if beta is not 0 or 1, this model may be unnormalized.
    target_lpdf = run(annealed_model.target; val=val)
    prior_lpdf  = run(annealed_model.prior; val=val)
    return annealed_model.beta * target_lpdf + (1 - annealed_model.beta) * prior_lpdf
end

function gradlogpdf(annealed_model::AnnealedModel, val)
    annealed_model.beta * gradlogpdf(annealed_model.target, val) + (1 - annealed_model.beta) * gradlogpdf(annealed_model.prior, val)
end

function run_ais(model, prior, kernel, annealing_schedule; return_trajectory=false)
    # The initial log weight is 0.0, because we sample from p_n exactly.
    x, = run(prior)
    weight = 0.0
    trajectory = [x]
    weights = [0.0]
    previous_model = AnnealedModel(model, prior, 0.0)
    for beta in annealing_schedule
        # Generate new sample
        current_model = AnnealedModel(model, prior, beta)

        # Score the sample
        weight += run(current_model; val=x) - run(previous_model; val=x)
        push!(weights, weight)

        # Update the sample
        for i in 1:1
            x, = run(kernel(current_model, x))
        end
        return_trajectory && push!(trajectory, x)
        previous_model = current_model
    end
    return_trajectory && return weight, weights, trajectory
    return weight
end

function geometric_schedule(first, last, N)
    R = (last/first)^(1.0/N)
    [first * R^i for i in 1:N]
end

# Compute an "ELBO" for AIS
function test_ais(mcvi, N, replicates; return_trajectories=false)
    # Use 20% of N on 0 to 0.1 linearly, then 80% on 0.1 to 1 geometrically
    N_linear = floor(N/5)
    N_linear_step = 0.005 / N_linear
    N_geom = N - N_linear
    # schedule = collect(0:(1/N):1.0)
    # schedule = geometric_schedule(1e-10, 1.0, N)
    schedule = [collect(N_linear_step:N_linear_step:0.005)..., geometric_schedule(0.005, 1.0, N_geom)...]
    runs = [run_ais(mcvi.model, mcvi.initial_proposal, mcvi.mcmc_kernel, schedule; return_trajectory=return_trajectories) for _ in 1:replicates]
    if return_trajectories
        return (ELBO=StatsBase.mean(map(first, runs)), Z=(logmeanexp(map(first, runs)), StatsBase.std(exp.(map(first, runs)))^2)), map(x -> x[2], runs), map(last, runs)
    else
        return (ELBO=StatsBase.mean(runs), Z=(logmeanexp(runs), StatsBase.std(exp.(runs))^2))
    end
end