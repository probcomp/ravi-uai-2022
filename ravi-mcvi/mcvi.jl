struct MCVIAlgorithm
    initial_proposal :: ProbModule
    model            :: ProbModule
    mcmc_kernel      :: Kernel
    metainference    :: MCMCMetaInference
end

struct MCMCTrajectoryDistribution <: ProbModule
    initial_proposal :: ProbModule
    step :: Kernel
    model :: ProbModule
    L :: Int
end

# Generate a trace of MCMC, or score one.
# If a trajectory is provided, the length 
# parameter is ignored.
function run(m::MCMCTrajectoryDistribution; val = nothing)
    if !isnothing(val)
        initial_weight = run(m.initial_proposal; val=val[1])
        return initial_weight + sum(run(m.step(m.model, val[i]); val=val[i+1]) for i in 1:m.L; init=0.0)
    end

    location, weight = run(m.initial_proposal)
    trajectory = [location]

    for i in 1:m.L
        location, weight_increment = run(m.step(m.model, location))
        push!(trajectory, location)
        weight += weight_increment
    end

    return trajectory, weight
end

function train!(m::MCVIAlgorithm, L, batch_size; epochs = 1, verbose=true)

    training_dist = MCMCTrajectoryDistribution(m.initial_proposal, m.mcmc_kernel, m.model, L)

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