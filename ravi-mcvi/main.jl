import Gen, BSON, StatsBase, Flux, Distributions, StatsPlots, Plots, Zygote, Dates, Serialization

include("../shared.jl")
include("prob_module.jl")
include("ula.jl")
include("nn_heads.jl")
include("sequential_meta_inference.jl")
include("step_embeddings_nn.jl")
include("mcvi.jl")
include("viz.jl")
include("smc.jl")
include("ais.jl")
include("ais-meta.jl")

# Toy target distributions
unimodal_model = DistributionModule(normal, (-1, 0.2))
multimodal_model = DistributionModule(
    HomogeneousMixture(Gen.Normal(), [0, 0]),
    ([0.5, 0.2, 0.3], [-3.0, 0.0, 2.0], [0.3, 1.0, 0.2])
)
experiment_names = ["unimodal-target", "multimodal-target"]
models = [unimodal_model, multimodal_model]
# MCMC settings
initial_proposals = [DistributionModule(normal, (0, 3)), DistributionModule(normal, (0, 3))]
kernels = [ULAKernel(0.015), ULAKernel(0.015)]

# Run some "warm-up chains" for each experimental setting,
# to fit empirical marginals for each time step of MCMC.
# These will be used to create the intermediate distributions
# for SMC meta-inference.
MAX_STEPS = 100
NUM_REPLICATES_FOR_MARGINAL_ESTIMATION = 100
trajectory_distributions = [MCMCTrajectoryDistribution(q, k, p, MAX_STEPS)
                            for (q, k, p) in zip(initial_proposals, kernels, models)]
example_chains = [[first(run(chain_distribution)) for _ in 1:NUM_REPLICATES_FOR_MARGINAL_ESTIMATION]
                  for chain_distribution in trajectory_distributions]
learned_marginals = [[DistributionModule(normal,
                        (StatsBase.mean(chain[step] for chain in chain_list), 
                         StatsBase.std(chain[step]  for chain in chain_list))) 
                      for step in 1:MAX_STEPS+1]
                      for chain_list in example_chains]

# Fitting meta-inference via variational training.
BATCH_SIZE = 200
TRAIN_EPOCHS = 100
#SAVED_WEIGHTS = ["simple_target_2022-01-28T16:03:12.568", "multimodal-target-2022-01-31T05:57:39.481"]
SAVED_WEIGHTS = ["unimodal-target-2022-08-09T15:17:04.764", "multimodal-target-2022-08-09T15:23:04.628"]
metainference_strategies = [step_embeddings_mlp(10, 1, [100, 100, 20, 2], GaussianHead()) for _ in 1:length(models)]
mcvi_algorithms = [MCVIAlgorithm(q, p, k, m) for (q, p, k, m) in zip(initial_proposals, models, kernels, metainference_strategies)]
for (weights_file, name, mcvi) in zip(SAVED_WEIGHTS, experiment_names, mcvi_algorithms)
    if isnothing(weights_file)
        train!(mcvi, MAX_STEPS, BATCH_SIZE; epochs=TRAIN_EPOCHS)
        save_weights!("../saved-weights/$(name)-$(Dates.now())", mcvi.metainference)
    else
        load_weights!("../saved-weights/$weights_file", mcvi.metainference)
    end
end
train!(mcvi_algorithms[1], 100, 200)


# Plot the example chains
fig1 = plot([], color="gray", legend=:bottomright, label="forward trajectories")
for chain in example_chains[2]
    plot!(chain, color="gray", alpha=0.3, ylims=(-10, 8), label=nothing)
end
plot!()
example_chains = [[first(run(chain_distribution)) for _ in 1:300]
                  for chain_distribution in trajectory_distributions]

backward_trajs = [[x, generate_backward_trajectory(metainference_strategies[2], x, 100)[1]...] for x in [last(c) for c in example_chains[2]]]
fig2 = plot([], legend=:bottomright, color="red", label="inferred backward trajectories (MCVI)")
for chain in backward_trajs
    plot!(reverse(chain), label=nothing, color="red", alpha=0.2, ylims=(-10, 8))
end
plot!()


function generate_backward_smc(x)
    particles, weights = smc(x, 
        mcvi_algorithms[2], 
        SMCParams(20, 100, 0.25, EmpiricalStepMarginals(learned_marginals[2], mcvi_algorithms[2].initial_proposal));
        retained=nothing, 
        return_particles=true,
        return_weight_histories=false)

        log_normalized_weights = weights .- logsumexp(weights)
        i = Gen.random(categorical, exp.(log_normalized_weights))
        particles[i]
end


backward_trajs_smc = [generate_backward_smc(x) for x in [last(c) for c in example_chains[2]]]
fig3 = plot([], legend=:bottomright, color="green", label="inferred backward trajectories (RAVI-MCVI)")
for chain in backward_trajs_smc
    plot!(reverse(chain), label=nothing, color="green", alpha=0.2, ylims=(-10, 8))#@, xlabel="# MCMC steps")
end
plot!()

plot(fig1, fig2, fig3, layout=(3,1))

savefig("../generated-figures/metainf-compare.pdf")



# animate_gaussian_metainference_dists(mcvi_algorithms[1].metainference; steps=[1, 50, 100], fps=1)
# Running the experiment:
#   * For each M, for each K, for each ESS threshold, estimate the ELBO 
#     and the variance of Z.
RESULTS_FILE = "mcmc-metainference-2022-01-31T21:45:17.092.log"
Ms = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
AIS_Ms = [10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
Ks = [1, 5, 10, 20, 50]
NUM_REPLICATES_FOR_ELBO_ESTIMATION = 2000
ess_thresholds = [0.0, 0.25]
if isnothing(RESULTS_FILE)
    elbos = Dict()
    zs = Dict()
else
    (elbos, zs) = Serialization.deserialize("../saved-results/$RESULTS_FILE")
end

for (name, mcvi, marginals) in zip(experiment_names, mcvi_algorithms, learned_marginals)
    # Main experiments: SMC
    for M in Ms
        for K in Ks
            for ess in ess_thresholds
                settings = (K=K, M=M, ess=ess, model=name)
                if haskey(elbos, settings) && haskey(zs, settings)
                    continue
                end

                smc_params = SMCParams(K, M, ess, EmpiricalStepMarginals(marginals, mcvi.initial_proposal))
                @time proposals = [mcmc_propose(mcvi, smc_params) for _ in 1:NUM_REPLICATES_FOR_ELBO_ESTIMATION]
                weights = [run(mcvi.model; val = loc) - q for (loc, q) in proposals]

                elbos[settings] = StatsBase.mean(weights)
                zs[settings] = (logmeanexp(weights), StatsBase.std(exp.(weights))^2)
                println("$settings — L=$(elbos[settings]), Z=$(zs[settings])")
            end
        end
    end

    # Using learned marginal distributions directly
    for M in Ms[2:end]
        settings = (baseline=:learned_marginals, M=M, model=name)
        if haskey(elbos, settings) && haskey(zs, settings)
            continue
        end
        marg = marginals[M]
        proposals = [run(marg) for _ in 1:NUM_REPLICATES_FOR_ELBO_ESTIMATION]
        weights = [run(mcvi.model; val = proposal) - q_weight for (proposal, q_weight) in proposals]
        elbos[settings] = StatsBase.mean(weights)
        zs[settings] = (logmeanexp(weights), StatsBase.std(exp.(weights))^2)
        println("$settings — L=$(elbos[settings]), Z=$(zs[settings])")
    end

    # Construct a stationary version of the algorithm, if necessary for AIS baseline
    if is_uncorrected(mcvi.mcmc_kernel)
        corrected_alg = MCVIAlgorithm(mcvi.initial_proposal, mcvi.model, corrected_version(mcvi.mcmc_kernel), mcvi.metainference)
    else
        corrected_alg = mcvi
    end

    # AIS baseline
    for M in AIS_Ms
        settings = (baseline=:ais, M=M, model=name)
        if haskey(elbos, settings) && haskey(zs, settings)
            continue
        end
        res = test_ais(corrected_alg, M, NUM_REPLICATES_FOR_ELBO_ESTIMATION)
        elbos[settings] = res.ELBO
        zs[settings] = res.Z 
        println("$settings — L=$(elbos[settings]), Z=$(zs[settings])")
    end
end
Serialization.serialize("../saved-results/mcmc-metainference-$(Dates.now()).log", (elbos, zs))


# Let's make some plots
COLORS = ["cyan", "green", "teal", "blue"]
MARKERS = [:x, :o, :+, :star4]
fig = plot(xlabel = "# MCMC steps", ylabel = "|log p(y) - L|", title="Unimodal target")
plot!(Ms, map(m -> -elbos[(K=1, M=m, ess=0.25, model="unimodal-target")], Ms), label="MCVI", color="red", marker=:star)
for (K, color, marker) in zip(Ks[2:end], COLORS, MARKERS)
    plot!(Ms, map(m -> -elbos[(K=K, M=m, ess=0.25, model="unimodal-target")], Ms), linewidth=2, marker=marker, color=color, label="RAVI-MCVI($K)", ylims=(0, 1.0))    
end
plot!()
savefig("../generated-figures/unimodal.pdf")

fig = plot(xlabel = "# MCMC steps", ylabel = "|log p(y) - L|", title="Unimodal target")
plot!(Ms, map(m -> -elbos[(K=1, M=m, ess=0.00, model="unimodal-target")], Ms), label="MCVI", color="red", marker=:star)
for (K, color, marker) in zip(Ks[2:end], COLORS, MARKERS)
    plot!(Ms, map(m -> -elbos[(K=K, M=m, ess=0.00, model="unimodal-target")], Ms), linewidth=2, marker=marker, color=color, label="RAVI-MCVI($K) (SIR)", ylims=(0, 1.0))    
end
plot!()
savefig("../generated-figures/unimodal-sir.pdf")


fig = plot(xlabel = "# MCMC steps", ylabel = "|log p(y) - L|", title="Multimodal target (SIR)")
plot!(Ms, map(m -> -elbos[(K=1, M=m, ess=0.00, model="multimodal-target")], Ms), label="MCVI", color="red", marker=:star)
for (K, color, marker) in zip(Ks[2:end], COLORS, MARKERS)
    plot!(Ms, map(m -> -elbos[(K=K, M=m, ess=0.00, model="multimodal-target")], Ms), linewidth=2, marker=marker, color=color, label="RAVI-MCVI($K)", ylims=(0, 5.0))    
end
plot!()
savefig("../generated-figures/multimodal-sir.pdf")


# Comparison to AIS
fig = plot(xlabel = "# MCMC steps", ylabel = "|log p(y) - L|", title="Multimodal target")
plot!(AIS_Ms, map(m -> -elbos[(baseline=:ais, M=m, model="multimodal-target")], AIS_Ms), label="AIS", marker=:star, linewidth=2)
# plot!(Ms, map(m -> -elbos[(K=1, M=m, ess=0.25, model="multimodal-target")], Ms), label="MCVI", color="red", marker=:star)
for (K, color, marker) in zip(Ks[2:end], COLORS, MARKERS)
    plot!(Ms .* K, map(m -> -elbos[(K=K, M=m, ess=0.25, model="multimodal-target")], Ms), linewidth=2, marker=marker, color=color, label="RAVI-MCVI($K)", ylims=(0, 5.0))    
end
plot!()
savefig("../generated-figures/Multimodal-AIS-comp.pdf")

fig = plot(xlabel = "# MCMC steps", ylabel = "|log p(y) - L|", title="Unimodal target")
plot!(AIS_Ms, map(m -> -elbos[(baseline=:ais, M=m, model="unimodal-target")], AIS_Ms), label="AIS", linewidth=2, marker=:star)
# plot!(Ms, map(m -> -elbos[(K=1, M=m, ess=0.25, model="multimodal-target")], Ms), label="MCVI", color="red", marker=:star)
for (K, color, marker) in zip(Ks[2:end], COLORS, MARKERS)
    plot!(Ms .* K, map(m -> -elbos[(K=K, M=m, ess=0.25, model="unimodal-target")], Ms), linewidth=2, marker=marker, color=color, label="RAVI-MCVI($K)", ylims=(0, 1.0))    
end
plot!()
savefig("../generated-figures/Unimodal-AIS-comp.pdf")