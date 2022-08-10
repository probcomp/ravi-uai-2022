struct AISTrajectoryDistribution <: ProbModule
    initial_proposal :: ProbModule
    step :: Kernel
    model :: ProbModule
    schedule :: Vector{Float64}
end

# Generate a trace of AIS, or score one.
# If a trajectory is provided, its length
# must match the schedule length.
function run(m::AISTrajectoryDistribution; val = nothing)
    annealed_dists = [AnnealedModel(m.model, m.initial_proposal, beta) for beta in m.schedule]

    if !isnothing(val)
        initial_weight = run(m.initial_proposal; val=val[1])
        return initial_weight + sum(run(m.step(model, val[i]); val=val[i+1]) for (i, model) in enumerate(annealed_dists[2:end]); init=0.0)
    end

    location, weight = run(m.initial_proposal)
    trajectory = [location]

    for model in annealed_dists
        location, weight_increment = run(m.step(model, location))
        push!(trajectory, location)
        weight += weight_increment
    end

    return trajectory, weight
end

function ais_train!(m::MCVIAlgorithm, schedule::Vector{Float64}, batch_size; epochs = 1, verbose=true)

    training_dist = AISTrajectoryDistribution(m.initial_proposal, m.mcmc_kernel, m.model, schedule)

    for epoch in 1:epochs

        # Generate batch of training examples
        trajectories = [first(run(training_dist)) for _ in 1:batch_size]

        # Use them to perform supervised meta-inference learning
        l = train!(m.metainference, trajectories)

        if verbose
            println("Epoch $epoch: loss=$l")
        end

    end

end

struct AnnealingSchedule <: Kernel
    schedule :: Vector{Float64}
    target :: ProbModule
    prior :: ProbModule
end
function (a::AnnealingSchedule)(which_step)
    AnnealedModel(a.target, a.prior, a.schedule[which_step])
end
function ais_propose(mcvi, schedule, K, ess_threshold=0.25; return_weight_histories=false, return_particles=false)
    history, weights = [], []

    annealed = AnnealingSchedule(schedule, mcvi.model, mcvi.initial_proposal)
    
    smc_options = SMCParams(K, length(schedule)-1, ess_threshold, annealed, true)

    # Initial location
    location, = run(mcvi.initial_proposal)
    push!(history, location)

    # Run MCMC
    for which_step = 2:smc_options.M+1
        location, weight = run(mcvi.mcmc_kernel(annealed(which_step), location))
        push!(history, location)
        push!(weights, weight)
    end

    # Run SMC
    final_location = pop!(history)
    retained = RetainedParticle(reverse(history), reverse(weights))
    final_location, smc(final_location, mcvi, smc_options; retained=retained, return_particles=return_particles, return_weight_histories=return_weight_histories)
end