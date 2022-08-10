include("prob_module.jl")

# A kernel, when applied to arguments, gives a 
# ProbModule.
abstract type Kernel end

struct ULAKernel <: Kernel
    step_size::Float64
end

function (k::ULAKernel)(model::ProbModule, val)
    std = sqrt(2 * k.step_size)
    grad = gradlogpdf(model, val)
    mean = val + k.step_size * grad
    return DistributionModule(normal, (mean, std))
end

is_uncorrected(::ULAKernel) = true
corrected_version(u::ULAKernel) = MALAKernel(u.step_size)

# MALA kernel
struct MALAKernel <: Kernel
    step_size::Float64
end

struct MALADist <: ProbModule
    model :: ProbModule
    step_size :: Float64
    initial_val
end
is_uncorrected(::MALAKernel) = false

function (k::MALAKernel)(model::ProbModule, val)
    MALADist(model, k.step_size, val)
end
function run(d::MALADist; val=nothing)
    if !isnothing(val)
        @error "Cannot analytically compute density of MALA kernel"
    end

    std = sqrt(2 * d.step_size)
    grad = gradlogpdf(d.model, d.initial_val)
    mean = d.initial_val + d.step_size * grad
    forward_dist = DistributionModule(normal, (mean, std))
    new_val, forward_weight = run(forward_dist)
    backward_grad = gradlogpdf(d.model, new_val)
    backward_mean = new_val + d.step_size * backward_grad
    backward_dist = DistributionModule(normal, (backward_mean, std))
    backward_weight = run(backward_dist; val=d.initial_val)

    model_score_old = run(d.model; val=d.initial_val)
    model_score_new = run(d.model; val=new_val)

    alpha = model_score_new + backward_weight - model_score_old - forward_weight
    if log(rand()) < alpha
        return new_val, NaN
    else
        return d.initial_val, NaN
    end
end


