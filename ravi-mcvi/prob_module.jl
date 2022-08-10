using Gen

abstract type ProbModule end

# Probabilistic module based on black-box implementations of 
# Propose and Assess. Does not support gradients.
struct ProposeAssessProbModule <: ProbModule
    propose :: Function
    assess  :: Function
end

function run(p::ProposeAssessProbModule; val=nothing)
    isnothing(val) && return p.propose()
    return p.assess(val)
end

function gradlogpdf(p::ProposeAssessProbModule, val)
    @error "Gradients not implemented for this probabilistic module"
end


# Probabilistic module based on a Gen generative function
# with a single traced random choice.
struct OneChoiceGenFn <: ProbModule
    gen_fn :: GenerativeFunction
    args :: Tuple
    addr
end

function run(g::OneChoiceGenFn; val=nothing)
    if isnothing(val)
        cm, w = Gen.propose(g.gen_fn, g.args)
        return cm[g.addr], w
    end

    first(Gen.assess(g.gen_fn, g.args, choicemap(g.addr=>val)))
end

function gradlogpdf(g::OneChoiceGenFn, val)
    trace = generate(g.gen_fn, g.args, choicemap(g.addr => val))
    retval_grad = accepts_output_grad(g.gen_fn) ? zero(get_retval(trace)) : nothing
    selection = select(g.addr)
    (_, values_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    return gradient_trie[g.addr]
end

# Probabilistic module based on a Gen distribution instantiated 
# with certain arguments.
struct DistributionModule <: ProbModule
    d :: Gen.Distribution
    args :: Tuple
end

function run(g::DistributionModule; val=nothing)
    should_sample = isnothing(val)
    
    if should_sample
        val = Gen.random(g.d, g.args...)
    end

    weight = Gen.logpdf(g.d, val, g.args...)

    return should_sample ? (val, weight) : weight
end

# Gradient with respect to output (the choice)
function gradlogpdf(g::DistributionModule, val)
    first(Gen.logpdf_grad(g.d, val, g.args...))
end
