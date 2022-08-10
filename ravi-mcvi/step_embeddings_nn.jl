# A type of neural backward proposal.
struct StepEmbeddingsNeuralProposal <: BackwardSequentialMetaInference
    step_embeddings::Matrix{Float64}
    head::Kernel
    body # Flux model
    optimizer # Flux optimizer
end

# Index of step being *predicted*, counting from 1 (start of chain)
# to T-1 (second-to-last element of chain)
function predict_previous_step(s::StepEmbeddingsNeuralProposal, next_loc, step_index::Int)
    nn_results = s.body(vcat(s.step_embeddings[step_index, :], Float32[next_loc]))
    s.head(nn_results)
end

function predict_previous_step_vectorized(s::StepEmbeddingsNeuralProposal, next_locs::Vector, step_indices)
    nn_results = s.body(transpose(hcat(s.step_embeddings[step_indices, :], next_locs)))
    s.head.(eachcol(nn_results))
end

function loss(s::StepEmbeddingsNeuralProposal, final_loc, backward_trajectory)
    L = length(backward_trajectory)
    next_locs = [final_loc, backward_trajectory[1:end-1]...]
    step_indices = L:-1:1
    predictions = predict_previous_step_vectorized(s, next_locs, step_indices)
    scores = [run(predictions[i]; val = backward_trajectory[i])
              for i in 1:L]
    return -sum(scores)
end

function save_weights!(name, s::StepEmbeddingsNeuralProposal)
    step_embeddings = s.step_embeddings
    model_params = Flux.params(s.body)
    BSON.@save "$(name)_step_embeddings.bson" step_embeddings
    BSON.@save "$(name)_model_params.bson" model_params
end

function load_weights!(name, s::StepEmbeddingsNeuralProposal)
    BSON.@load "$(name)_step_embeddings.bson" step_embeddings
    BSON.@load "$(name)_model_params.bson" model_params
    copy!(s.step_embeddings, step_embeddings)
    Flux.loadparams!(s.body, model_params)
end

# Question: should train! be different for each type of network?
# It seems like we could also just ask for the Flux parameters.
function trainable_flux_params(s::StepEmbeddingsNeuralProposal)
    return Flux.params(s.body, s.step_embeddings)
end

function flux_optimizer(s::StepEmbeddingsNeuralProposal)
    return s.optimizer
end


# Some sensible defaults
function step_embeddings_mlp(embedding_dim, data_dim,
    layer_dims, head;
    L = 100, opt = Flux.ADAM(1e-4))

    step_embeddings = randn(L, embedding_dim)

    num_layers = length(layer_dims)
    layer_dims = [embedding_dim + data_dim, layer_dims...]
    layers = [Flux.Dense(layer_dims[i], layer_dims[i+1],
        (i == num_layers) ? identity : Flux.relu)
              for i in 1:num_layers]
    net = Flux.Chain(layers...)

    StepEmbeddingsNeuralProposal(step_embeddings, head, net, opt)
end


struct EmpiricalStepMarginals <: Kernel
    marginals::Vector{ProbModule}
    exact::ProbModule
end
(e::EmpiricalStepMarginals)(which_step) = (which_step == 1 ? e.exact : e.marginals[which_step])
