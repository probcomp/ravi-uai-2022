struct GaussianHead <: Kernel end
(::GaussianHead)(nn_output) = DistributionModule(normal, (nn_output[1], exp(nn_output[2])))

struct MixtureOfGaussiansHead <: Kernel end
(::MixtureOfGaussiansHead)(nn_output) =
    ProposeAssessProbModule(
        () -> begin 
            p = Flux.sigmoid(nn_output[1])
            if rand() < p
                val = Gen.random(normal, nn_output[2], exp(nn_output[4]))
            else
                val = Gen.random(normal, nn_output[3], exp(nn_output[5]))
            end
            weight = Gen.logsumexp(log(p) + logpdf(normal, val, nn_output[2], exp(nn_output[4])),
                log1p(-p) + logpdf(normal, val, nn_output[3], exp(nn_output[5])))
            return (val, weight)
        end,
        val -> begin
            p = Flux.sigmoid(nn_output[1])
            Gen.logsumexp(log(p) + logpdf(normal, val, nn_output[2], exp(nn_output[4])),
                log1p(-p) + logpdf(normal, val, nn_output[3], exp(nn_output[5]))) 
        end
    )
