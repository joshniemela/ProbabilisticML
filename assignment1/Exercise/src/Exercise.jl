using Distributions
using LinearAlgebra

# A begin
px = MultivariateNormal([0, 0], [1 0; 0 1])
x = rand(px, 20)

# We fix theta
θ = [-1, 1]

y = Normal.(x' * θ, 0.1) .|> rand

dataset = (x, y)
# A end

#=
p(y | x, θ) is given
p(θ) is given

P(θ | x, y) = p(x, y | θ) * p(θ) / p(x, y)
=#
# B begin
prior_θ = MultivariateNormal([0, 0], [1 0; 0 1])

# This assumes a prior of mean 0 and variance 1.
function parameter_estimation(dataset)
    x, y = dataset
    k, l = size(x)

    μ = x * inv(var(y) * I(l) + x' * x) * y
    Σ = I(k) - x * inv(var(y) * I(l) + x' * x) * x'

    return μ, Σ
end
