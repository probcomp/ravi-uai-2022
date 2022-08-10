abstract type MCMCMetaInference end

# One kind of meta-inference uses sequential proposals 
# backward in time. These support SMC meta-inference.
abstract type BackwardSequentialMetaInference <: MCMCMetaInference end

# Abstract function to predict previous step
function predict_previous_step(b::BackwardSequentialMetaInference, next_loc, step_index) end

# Fallback if vectorized version is not implemented
function predict_previous_step_vectorized(b::BackwardSequentialMetaInference, next_locs, step_indices)
    @warn "Using unvectorized loss."

    return [predict_previous_step(b, next_loc, step_index) for (next_loc, step_index) in zip(next_locs, step_indices)]
end

function generate_backward_trajectory(b::BackwardSequentialMetaInference, final_loc, L)
    trajectory = []
    score = 0.0
    next_loc = final_loc
    for step_index in L:-1:1
        next_loc, weight = run(predict_previous_step(b, next_loc, step_index))
        push!(trajectory, next_loc)
        score += weight
    end
    return trajectory, score
end

function loss(b::BackwardSequentialMetaInference, final_loc, backward_trajectory)
    L = length(backward_trajectory)
    next_locs = [final_loc, backward_trajectory[1:end-1]...]
    step_indices = L:-1:1
    predictions = run.(predict_previous_step_vectorized(b, next_locs, step_indices))
    scores = [run(prediction; val = target)
              for (prediction, target) in zip(predictions, backward_trajectory)]
    return -sum(scores)
end

function train!(b::BackwardSequentialMetaInference, trajectories)
    params_to_train = trainable_flux_params(b)

    training_loss(trajectory) = loss(b, last(trajectory), reverse(trajectory[1:end-1]))

    Flux.train!(training_loss, params_to_train, trajectories, flux_optimizer(b))

    # Evaluate loss and return
    return (StatsBase.mean(training_loss(t) for t in trajectories))
end


# Given a final location, and a number of steps L,
# generate L steps "backward in time" and return 
# the proposed backward trajectory.
struct BackwardTrajectoryGenerator{S} <: Kernel
    proposal :: BackwardSequentialMetaInference
end

function (b::BackwardTrajectoryGenerator)(final_loc, L)
    ProposeAssessProbModule(
        # Propose
        () -> generate_backward_trajectory(b, final_loc, L),
        # Assess
        trajectory -> -loss(b, final_loc, trajectory)
    )
end
