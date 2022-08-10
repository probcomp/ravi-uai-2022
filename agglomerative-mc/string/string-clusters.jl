using StatsBase
using Distributions: logpdf, NegativeBinomial
using StringDistances

struct StringHyperparameters <: Hyperparameters
    options :: Vector{String}
    priors  :: Vector{Float64}
end

mutable struct StringStatistics <: ClusterStats
    likelihoods :: Vector{Float64}
end

copystats(stats::StringStatistics) = StringStatistics(copy(stats.likelihoods))

const StringCluster = Cluster{StringStatistics, StringHyperparameters}
const StringTrace = DPMMTrace{StringCluster, StringHyperparameters}

function empty_cluster_stats(trace::StringTrace)
    StringStatistics(zeros(length(trace.hypers.options)))
end

function update_stats_after_adding_point!(cluster::StringCluster, x::String)
    options = cluster.trace.hypers.options
    cluster.stats.likelihoods += typos_density.(options, x)
end

function update_stats_after_removing_point!(cluster::StringCluster, x::String)
    options = cluster.trace.hypers.options
    cluster.stats.likelihoods -= typos_density.(options, x)
end

function conditional_likelihood(trace::StringTrace, cluster_id::Symbol, provisional_cluster::StringCluster)
    hypers = trace.hypers
    existing_cluster = get(trace.clusters, cluster_id) do 
        Cluster(trace, Set{Int}(), empty_cluster_stats(trace))
    end

    current_joints = existing_cluster.stats.likelihoods .+ hypers.priors
    updated_priors = current_joints .- logsumexp(current_joints)
    logsumexp(provisional_cluster.stats.likelihoods .+ updated_priors)
end


# Update stats of existing cluster to include new datapoints
function merge_stats!(trace::StringTrace, existing_cluster_id::Symbol, new_cluster::StringCluster)
    cluster = trace.clusters[existing_cluster_id]
    cluster.stats.likelihoods .+= new_cluster.stats.likelihoods
end


function summarize(trace::StringTrace)
    [(length(cluster.members), trace.data[sort(collect(cluster.members))]) for (_, cluster) in trace.clusters]
end


function singleton_cluster(trace::StringTrace, index::Int)
    liks = typos_density.(trace.hypers.options, trace.data[index])
    Cluster{StringStatistics, StringHyperparameters}(trace, Set(index), StringStatistics(liks))
end

letter_probs_file = "../../datasets/letter_probabilities.csv"
letter_trans_file = "../../datasets/letter_transition_matrix.csv"
const initial_letter_probs = CSV.File(letter_probs_file; header=false) |> DataFrame |> Array{Float64}
const english_letter_transitions = CSV.File(letter_trans_file; header=false) |> DataFrame |> Matrix{Float64}
const alphabet = [collect('a':'z')..., ' ', '.']
const alphabet_lookup = Dict([l => i for (i, l) in enumerate(alphabet)])
const UNUSUAL_LETTER_PENALTY = 1000
min_length = 40
max_length = 200

function string_prior_density(observed)
    if length(observed) < min_length || length(observed) > max_length
        return -Inf
    end
    score = -log(max_length-min_length+1)
    if length(observed) == 0
        return score
    end

    prev_letter = nothing
    for letter in observed
        dist = isnothing(prev_letter) ? initial_letter_probs : vec(english_letter_transitions[:, prev_letter])
        prev_letter = haskey(alphabet_lookup, lowercase(letter)) ? alphabet_lookup[lowercase(letter)] : nothing
        score += isnothing(prev_letter) ? -log(28) : max(log(dist[prev_letter]), -UNUSUAL_LETTER_PENALTY)
    end
    score
end
    

const add_typos_density_dict = Dict{Tuple{String, String}, Float64}()
const LETTERS_PER_TYPO = 5.0
using StringDistances
using Distributions: NegativeBinomial
typos_density(word::String, observed::String) = begin 
    get!(add_typos_density_dict, (observed, word)) do
    num_typos = evaluate(DamerauLevenshtein(), observed, word)
    
    l = logpdf(NegativeBinomial(ceil(length(word) / LETTERS_PER_TYPO), 0.9), num_typos)
    l -= log(length(word)) * num_typos
    l -= log(26) * (num_typos) / 2 # Maybe we should actually compute the prob of the most probable typo path.
    l
    end
end
    
function make_hypers(options::Vector{String})
    dummy = ""
    for _ in 1:mean(length(s) for s in options)
        dummy = "$dummy*"
    end
    
    priors = [string_prior_density(s) for s in options]
    tot = logsumexp(priors)
    remainder = log1p(-exp(tot))
    push!(options, dummy)
    push!(priors, remainder)
    return StringHyperparameters(options, priors)
end