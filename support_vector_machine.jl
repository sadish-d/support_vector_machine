
# VERSION == v"1.8.5" #= Julia =#

module SupportVectorMachine

# types and structs
export Problem
export BiClass

# functions
export feature_vectors
export feature_vectors!
export labels
export labels!
export weights
export weights!
export scaler
export scaler!
export predictions
export predictions!

export validate_feature_vectors
export validate_labels
export validate_weights
export validate_scaler
export validate_BiClass

export loss_hinge_regularized
export gradient_hinge_regularized

export gradient_descent!

export predict!
export hyperplane_points

abstract type Problem end
mutable struct BiClass<:Problem # binary classification
    feature_vectors::Vector{Vector{<:Real}} # vector of xs
    labels::Vector{Int64} # vector of ys
    scaler::Real # λ (Lagrange multiplier)
    weights::Vector{<:Real} # w
    predictions::Vector{Int64} # vector of predicted ys

    function BiClass(feature_vectors, labels, scaler, weights, predictions)
        # validate arguments
        isempty(labels) || length(feature_vectors) == length(labels) ||
            error("Expected the same number of feature vectors as number of labels.")
        issubset(Set(labels), Set((-1, 1))) ||
            error("Labels must be -1 or 1.")
        scaler > 0 ||
            error("Expected the scalar (Lagrange multiplier) to be positive.")
        isempty(weights) || [length(weights) == length(v) for v in feature_vectors] |> all ||
            error("Feature vectors and weights do not have the same length.")
        [v[1]==one(Real) for v in feature_vectors] |> all ||
            error("The first entry in all feature vectors must be one.")
        isempty(predictions) || length(feature_vectors) == length(predictions) ||
            error("Expected the same number of feature vectors as number of predictions.")
                
        return new(feature_vectors, labels, scaler, weights, predictions)
    end
    BiClass(feature_vectors, labels, scaler, weights)   = BiClass(feature_vectors, labels,  scaler, weights, Real[])
    BiClass(feature_vectors, labels, scaler)            = BiClass(feature_vectors, labels,  scaler, Real[],  Real[])
    BiClass(feature_vectors, labels)                    = BiClass(feature_vectors, labels,  1.0,    Real[],  Real[])
    BiClass(feature_vectors)                            = BiClass(feature_vectors, Int64[], 1.0,    Real[],  Real[])
    BiClass()                                           = BiClass(Vector{Real}[],  Int64[],  1.0,   Real[],  Real[])
end

feature_vectors(problem::BiClass) = problem.feature_vectors

function feature_vectors!(problem::BiClass, input::Vector{Vector{<:Real}})
    problem.feature_vectors = input
    return nothing
end

labels(problem::BiClass) = problem.labels

function labels!(problem::BiClass, input::Vector{Int64})
    issubset(Set(labels(problem)), Set((-1, 1))) || error("Labels must be -1 or 1.")
    problem.labels = input
    return nothing
end

scaler(problem::BiClass) = problem.scaler

function scaler!(problem::BiClass, input::Real)
    problem.scaler = input
    return nothing
end

weights(problem::BiClass) = problem.weights

function weights!(problem::BiClass, input::Vector{<:Real})
    problem.weights = input
    return nothing
end

predictions(problem::BiClass) = problem.predictions

function predictions!(problem::BiClass, input::Vector{Int64})
    issubset(Set(predictions(problem)), Set((-1, 1))) || error("Predictions must be -1 or 1.")
    problem.predictions = input
    return nothing
end


# -------------------------------------------------------------
# functions for validiting Problem arguments
function validate_feature_vectors(problem::Problem)
    isempty(labels(problem)) || length(feature_vectors(problem)) == length(labels(problem)) || 
        error("Expected the same number of feature vectors as labels.")
    [v[1]==one(Real) for v in feature_vectors(problem)] |> all || 
        error("The first entry in all feature vectors must be one.")
end

function validate_labels(problem::Problem)
    issubset(Set(labels(problem)), Set((-1, 1))) || 
        error("Labels must be -1 or 1.")
end

function validate_scaler(problem::Problem)
    scaler(problem) > 0 || 
        error("Expected the scalar (Lagrange multiplier) to be positive.")
end

function validate_weights(problem::Problem)
    isempty(weights(problem)) || [length(weights(problem)) == length(v) for v in feature_vectors(problem)] |> all || 
        error("Feature vectors and weights do not have the same length.")
end

function validate_predictions(problem::Problem)
    issubset(Set(predictions(problem)), Set((-1, 1))) || 
        error("Predictions must be -1 or 1.")
end

function validate_BiClass(problem::BiClass)
    validate_feature_vectors(problem)
    validate_labels(problem)
    validate_scaler(problem)
    validate_weights(problem)
    validate_predictions(problem)
end
# -------------------------------------------------------------


"""
    regularized hinge loss: ∑ max(0, 1-ywᵀx) + λ‖w‖²/2

    The Lagrange multiplier λ (scaler) is a constant that determines
      the importance of regularization relative to that of hinge loss.
"""
function loss_hinge_regularized(problem::BiClass)    
    validate_BiClass(problem)
    
    regularizer = (weights(problem)' * weights(problem)) / 2    # ‖w‖²/2
    hinge_loss(x, y, w) = 1 - y * (w' * x) |>                   # for one observation: 1-ywᵀx
                            o-> max(0, o)                       # for one observation: max(0, 1-ywᵀx)
    hinge_loss_sum = [
        hinge_loss(x, y, weights(problem)) for
        (x, y) in zip(feature_vectors(problem), labels(problem))
    ] |> sum                                                    # sum of loss of all observations: ∑ max(0, 1-ywᵀx)
    loss = hinge_loss_sum + scaler(problem) * regularizer
    return loss
end

"""
    gradient of regularized hinge loss:
    λw + ∑-yx if y(wᵀx)<1
    λw        if y(wᵀx)≥1
"""
function gradient_hinge_regularized(problem::BiClass)
    validate_BiClass(problem)

    have_nonzero_grads = [
        (x, y) for (x, y) in
        zip(feature_vectors(problem), labels(problem))     # observations with non-zero hinge loss gradients
        if (y * (weights(problem)' * x) < 1)               # Hinge loss has gradient 0 for y(wᵀx)>=1.
    ]
    gradient_hinge_sum =
        if isempty(have_nonzero_grads)
            zeros(Float64, length(weights(problem)))       # If no feature vectors have non-zero gradient,
                                                           #   fetch a zero vector.
        else
            [(-y * x) for (x, y) in have_nonzero_grads] |> # Hinge loss has gradient -yx for ywᵀx<1.
            sum
        end
    gradient_regularizer = scaler(problem) * weights(problem)
    gradient = gradient_hinge_sum + gradient_regularizer
    return gradient
end

"""
    gradient descent
"""
function gradient_descent!(
    problem::Problem,
    loss_function,
    gradient_function,
    tolerance = 1e-6,
    max_iter = 2000,
    )
    validate_BiClass(problem)
    
    # initialize weights
    length(feature_vectors(problem)[1]) |> o-> weights!(problem, zeros(Float64, o))

    """
        line search function
        It ensures that the step in gradient descent is not so large that
          the gradient begins to increase.
    """
    function line_search(problem, loss_function, gradient, weights)
        learn_rate = 1; loss = 0; loss_new = 1      # to initialize loop
        while loss_new > loss
            learn_rate = learn_rate/2
            loss = loss_function(problem)
            weights_new = weights - learn_rate * gradient
            weights!(problem, weights_new)
            loss_new = loss_function(problem)
            weights!(problem, weights)              # We are not changing weights here. Set them back.
        end
        return learn_rate
    end

    gradient = tolerance # initialize
    iteration = 0
    while any(abs.(gradient) .>= tolerance) && iteration < max_iter
        gradient = gradient_function(problem)
        learn_rate = line_search(problem, loss_function, gradient, weights(problem))
        weights(problem) - learn_rate * gradient |> o-> weights!(problem, o)
        iteration += 1
        (iteration % 100 == 0) && println("iterations: ", iteration)
    end
end

function predict!(problem::BiClass)
    validate_BiClass(problem)

    y = [weights(problem)' * x for x in feature_vectors(problem)] |> 
        o-> [(score >=0) ? 1 : -1 for score in o]
    predictions!(problem, y)
    return nothing
end

"""
    For a hyperplane described by a normal vector and bias, get points on the
      hyperplane.

    Suppose w = [w0, w1, w2], w0 being the bias, and [w1, w2] the normal vector.
    We want an equation for all point [x1, x2] that lie on the hyperplane.
    We take x = [1, x1, x2], and solve for x such that wᵀx=0.
      wᵀx = 0
      [w0, w1, w2]ᵀ[1, x1, x2] = 0
      w0 + w1x1 + w2x2 = 0
      x2 = -(w0 + w1x1)/w2
    So for any value of x1, we know what x2 makes the point lie on the hyperplane.
"""
function hyperplane_points(weights::Vector{<:Real}, dimensions::Vector{T} where {T<:Vector{<:Real}})
    # The first element in the weights vector is the bias.
    bias = weights[begin]
    weights_and_dims = zip(weights[2:(end-1)], dimensions)  # removing from weights the bias (first element of weights) and
                                                            #     the weight of the dimension to be calculated (last weight).
    
    last_coordinate = - (bias .+ sum([weight * dim for (weight, dim) in weights_and_dims])) / weights[end]
    # coordinates = [                          # Promote allows appending when the elements of dimensions and
    #     append!(promote(ds, [ld])...)        #     last_coordinates are not the same.
    #     for ds in dimensions
    #     for ld in last_coordinate
    # ]
    coordinates = [vcat(ds, [ld]) for ds in dimensions for ld in last_coordinate]
    return coordinates
end

end #= end of module =#

#=
# -------------------------------------------------------------
# Example
# -------------------------------------------------------------
using .SupportVectorMachine

colors_binary(categories::Vector{Int64}, alpha=0.8) = [(c == 1) ? RGBAf(1, 0, 1, alpha) : RGBAf(0, 1, 1, alpha) for c in categories]


# creating a dataset that is seperated by the hyperplane [0, -1, 1], except for errors.
_xs = [
    [1, 1, 2],
    [1, 2, 1],
    [1, 3, 4],
    [1, 4, 3],
    [1, 5, 6],
    [1, 6, 5],
    [1, 7, 8],
    [1, 8, 7],
]
_y = [
     1
    -1
    -1 # the hyperplane [0, -1, 1] would separate the data were it not for this exception
    -1
     1
    -1
     1
    -1
]

_problem = BiClass(_xs, _y)

gradient_descent!(_problem, loss_hinge_regularized, gradient_hinge_regularized)

predict!(_problem)

# decision_boundary = hyperplane_points(weights(_problem), [collect(0:0.01:1)])

# create prediction dataset
_n  = 10
_n² = _n ^ 2
prediction_data = let
    _xs = Iterators.product(1:0.2:_n, 1:0.2:_n) .|> collect |>vec
    _xs1 = [vcat([1.0], x) for x in _xs]
    _xs1
end

prediction_problem = BiClass(prediction_data)
weights!(prediction_problem, weights(_problem))

predict!(prediction_problem)
# -------------------------------------------------------------
# -------------------------------------------------------------
=#

#=
# plots for the example
using CairoMakie

_training = scatter(
    [v[2] for v in feature_vectors(_problem)],
    [v[3] for v in feature_vectors(_problem)],
    color = colors_binary(predictions(_problem)),
    # marker = :dtriangle,
    # rotations = pi * [(l == -1) ? 0 : l for l in labels(_problem)] # actual data
    # markersize = 24,
    marker = :hline,
    rotations = (pi / 2) * [(l == -1) ? 0 : l for l in labels(_problem)],
    markersize = 56,
)
display(_training)
save("categories_actual_predicted.png", _training)


_model = scatter(
    [v[2] for v in feature_vectors(prediction_problem)],
    [v[3] for v in feature_vectors(prediction_problem)],
    color = colors_binary(predictions(prediction_problem)),
)
display(_model)
save("categories_model.png", _model)
=#

#=
# -------------------------------------------------------------
# Another exmaple
# -------------------------------------------------------------
using Random
# using CairoMakie # import if not already imported

# create data
n = 10_000
Random.seed!(947174)
points = rand(Float64, n, 3)
xyz = eachslice(points, dims=1) |> collect
Random.seed!(520784)
err = rand(Float64, n, 1) .- (1/2)
cat = [(point[1] + point[2] + e >= point[3]) ? 1 : -1 for (point, e) in zip(xyz, err)]

problem = BiClass(
    [vcat([1], point) for point in xyz],
    cat,
    )
scaler!(problem, 0.02) # extent of regularization

gradient_descent!(problem, loss_hinge_regularized, gradient_hinge_regularized)

predict!(problem)

# plot to show how model classifies training data after training
fig = Figure()
ax1 = Axis3(fig[1, 1],
           xlabel = "",
           ylabel = "",
           zlabel = "",
           aspect = (1, 1, 1),
           title="data",
)
ax2 = Axis3(fig[1, 2],
           xlabel = "",
           ylabel = "",
           zlabel = "",
           aspect = (1, 1, 1),
           title="model",
)
# data
scatter!(
    ax1,
    points[:, 1],
    points[:, 2],
    points[:, 3],
    markersize=6,
    color=colors_binary(labels(problem)),
)
# prediction
scatter!(
    ax2,
    points[:, 1],
    points[:, 2],
    points[:, 3],
    markersize=6,
    color=colors_binary(predictions(problem)),
    alpha=0.66,
)
fig
record(fig, "categories.gif", 1:120) do frame
    ax1.azimuth[] = 1.7pi + 0.3 * sin(2pi * frame / 120)
    ax2.azimuth[] = 1.7pi + 0.3 * sin(2pi * frame / 120)
end

# decision boundary
# hypl = hyperplane_points(weights(post_problem), [[x1, x2] for x1 in collect(0:0.01:1) for x2 in collect(0:0.01:1)])

using GLMakie
# meshscatter(points, marker=Sphere(Point3f(0), 1f0), markersize=0.01, color=cat)
# meshscatter(points[:, 1], points[:, 2], points[:, 3], marker=Sphere(Point3f(0), 1f0), markersize=0.01, color=cat)
# data
meshscatter(
    points[:, 1],
    points[:, 2],
    points[:, 3],
    markersize=0.01,
    color=colors_binary(labels(problem))
)
# prediction
meshscatter(
    points[:, 1],
    points[:, 2],
    points[:, 3],
    markersize=0.01,
    color=colors_binary(predictions(problem))
)

# -------------------------------------------------------------
# -------------------------------------------------------------
=#

