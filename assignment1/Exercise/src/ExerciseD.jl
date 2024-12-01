using Distributions
using LinearAlgebra
using GLMakie


# These functions come from the proofs in the lecture notes about GLM (week 1)
# This assumes a prior of mean 0 and variance 1.
function parameter_estimation(dataset)
    x, y = dataset
    k, l = size(x)

    μ = x * inv(var(y) * I(l) + x' * x) * y
    Σ = I(k) - x * inv(var(y) * I(l) + x' * x) * x'

    return μ, Σ
end

function posterior_predictive(x, dataset)
    _, y = dataset
    μ, Σ = parameter_estimation(dataset)

    Normal(x' * μ, var(y) + x' * Σ * x)
end

# A begin
px = MultivariateNormal([0, 0], [0.1 0; 0 1])
x = rand(px, 20)

# We fix theta
θ = [-1, 1]

y = Normal.(x' * θ, 0.1) .|> rand

dataset = (x, y)
# A end

# B begin
prior_θ = MultivariateNormal([0, 0], [1 0; 0 1])


μ, Σ = parameter_estimation(dataset)

pθ = MultivariateNormal(μ, Symmetric(Σ))

f(x, y) = pdf(pθ, [x; y])
xs = LinRange(-3, 3, 200)
ys = LinRange(-3, 3, 200)
zs = [f(x, y) for x in xs, y in ys]

contour(xs, ys, zs, levels=10)

# B end

# C begin
x1s = LinRange(-3, 3, 200)
x2s = LinRange(-3, 3, 200)
zs = [var(posterior_predictive([x1, x2], dataset)) for x1 in x1s, x2 in x2s]

contour(x1s, x2s, zs, levels=10)
# C end
