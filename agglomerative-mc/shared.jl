using Gen: logsumexp

function logmeanexp(weights)
    logsumexp(weights) - log(length(weights))
end

function remove_missing(v::Vector{Union{Missing, T}}) where T
    T[filter(x -> !ismissing(x), v)...]
end
function remove_missing(v::Vector{Missing})
    []
end
function remove_missing(v::Vector{T}) where T
    v
end

function normalize_logprobs(l)
    l .- logsumexp(l)
end