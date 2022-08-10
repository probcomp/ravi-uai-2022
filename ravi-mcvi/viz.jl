# Plots we might want:
#   - model density vs. 
#   - simulated trajectories and reverse trajectories, e.g. from SMC
#   - plots of inference quality over time for multiple algorithms
using Plots

# Assumes head is Gaussian!
function animate_gaussian_metainference_dists(b::BackwardSequentialMetaInference; steps=1:100, fps=5, test_xs=-5:0.1:5)
    vis = @animate for i in steps
        test_ys =  [predict_previous_step(b, x, i).args[1] for x in test_xs]
        test_std = [predict_previous_step(b, x, i).args[2] for x in test_xs]
    
        plot(Plots.Shape([[(x,y+std) for (x, y, std) in zip(test_xs, test_ys, test_std)]..., 
                          [(x,y-std) for (x, y, std) in reverse(collect(zip(test_xs, test_ys, test_std)))]...]), 
                          fill="gray", alpha=0.4, label=nothing, ylims=[-5, 5],
                          xlabel="Step $(i+1)", ylabel="Predicted value of Step $i")

        plot!(test_xs, test_ys, color="teal", label=nothing)
        plot!(test_xs, test_xs, color="black", label=nothing)
    end
    gif(vis, fps=fps)
end
