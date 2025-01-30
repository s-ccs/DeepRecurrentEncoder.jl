"""
    mse(y_pred, y_true)

Calculate the mean squared error (MSE) between predicted and true values.

# Arguments
- `y_pred::AbstractArray`: Predicted values.
- `y_true::AbstractArray`: True values.

# Returns
- `Float64`: The computed MSE.
"""
function mse(y_pred, y_true)
    return sum((y_pred .- y_true) .^ 2)
end

"""
    r_squared(y_pred, y_true)

Compute the coefficient of determination (R²) for the given predictions.

# Arguments
- `y_pred::AbstractArray`: Predicted values.
- `y_true::AbstractArray`: True values.

# Returns
- `Float64`: The R² score.
"""
function r_squared(y_pred, y_true)
    y_mean = sum(y_true) / length(y_true)
    ss_tot = sum((y_true .- y_mean) .^ 2)
    ss_res = sum((y_true .- y_pred) .^ 2)
    return 1 - ss_res / ss_tot
end

"""
    compute_loss(x, y, model, ps, st; loss_function=mse)

Compute the loss for the given inputs, model, and loss function.

# Arguments
- `x`: Input data.
- `y`: Ground truth values.
- `model`: Function representing the predictive model.
- `ps`: Model parameters.
- `st`: Model state.
- `loss_function`: Loss function to use (default: `mse`).

# Returns
- `(Float64, AbstractArray, Any)`: Tuple of loss, predicted values, and updated model state.
"""
function compute_loss(x, y, model, ps, st, loss_function=mse)
    y_pred, st = model(x, ps, st)
    return loss_function(y_pred, y), y_pred, st
end

"""
    add_mask(eeg, p)

Add a mask to the EEG data. This is done to mask any missing data in EEG recordings and maintain data consistancy.
EEG recordings are typically inconsistant in the channels dimention. This may be due to errors in measurement or equipment misfunction. Hence, we mask a percentage of the channels to reduce the interference of this data in the final model.

# Arguments
- `eeg::AbstractArray`: The input EEG data of shape `(channels, time, epochs)`.
- `p::Float64`: The proportion of data to mask.

# Returns
- `AbstractArray`: A new array containing the EEG data with a mask added.
"""
function add_mask(eeg::AbstractArray, p)
    masked = similar(eeg, size(eeg, 1), size(eeg, 2) + 1, size(eeg, 3))
    n_channels = size(eeg, 2)
    masked[:, 1:n_channels, :] .= eeg
    add_mask!(masked, eeg, p)
end

"""
    add_mask!(masked, eeg, p)

In-place version of `add_mask`.

# Arguments
- `masked::AbstractArray{T,3}`: Array to store masked EEG data.
- `eeg::AbstractArray{T,3}`: Input EEG data of shape `(channels, time, epochs)`.
- `p::Float64`: Proportion of data to mask.

# Returns
- `AbstractArray`: The masked array.
"""
function add_mask!(masked::AbstractArray{T,3}, eeg::AbstractArray{T,3}, p) where {T}
    index_mask_to = Int(floor(size(eeg, 1) * p))

    masked[index_mask_to:end, :, :] .= 0
    masked[1:index_mask_to, end, :] .= 0
    masked[index_mask_to:end, end, :] .= 1

    return masked
end

"""
    generate_designmatrix(f, evts)

Generate the design matrix for given formula and events. This is to convert a dataset and a formula into a numerical design matrix, making it easier to fit statistical or machine learning models

# Arguments
- `f::FormulaTerm`: A formula describing the design.
- `evts`: Events data.

# Returns
- `AbstractArray`: The design matrix.
"""
function generate_designmatrix(f::FormulaTerm, evts)
    f = apply_schema(f, schema(f, evts))
    _, X = modelcols(f, evts)
    return X
end

"""
    add_stimuli(in, designmatrix)

Add stimuli information to EEG data. Used to combine EEG signals with stimuli data from the design matrix.

# Arguments
- `in::AbstractArray{T,3}`: EEG data of shape `(time, channels, epochs)`.
- `designmatrix::AbstractArray{T,2}`: Matrix describing stimuli for each epoch.

# Returns
- `AbstractArray`: EEG data with stimuli added.
"""
function add_stimuli(in::AbstractArray{T,3}, designmatrix::AbstractArray{T,2}) where {T}
    n_chan = size(in, 2)
    n_time = size(in, 1)
    n_rep = size(in, 3)
    out = similar(designmatrix, n_time, n_chan + size(designmatrix, 2), n_rep)
    @debug typeof(in), typeof(out)
    out[:, 1:n_chan, :] .= Array(in)
    add_stimuli!(out, in, designmatrix)
    return out
end

"""
    add_stimuli!(out, in, designmatrix)

In-place version of `add_stimuli`.

# Arguments
- `out`: Output array to store the result.
- `in`: EEG data of shape `(time, channels, epochs)`.
- `designmatrix`: Matrix describing stimuli for each epoch.

# Returns
- `AbstractArray`: The output array with stimuli added.
"""
function add_stimuli!(out, in, designmatrix)
    @debug size(out), size(in), size(designmatrix)
    for c = axes(designmatrix, 2)
        for e = axes(designmatrix, 1)
            for t = axes(out, 1)
                out[t, size(in, 2)+1:end, e] .= designmatrix[e, c]
            end
        end
    end
    return out
end

"""
    train_test_split(data, evts; at=0.7)

Split data and events into training and testing sets.

# Arguments
- `data::AbstractArray`: Input data of shape `(time, channels, epochs)`.
- `evts::AbstractArray`: Events data.
- `at::Float64`: Proportion of data to include in the training set (default: `0.7`).

# Returns
- `(AbstractArray, AbstractArray, AbstractArray, AbstractArray)`: Training data, training events, testing data, and testing events.
"""
function train_test_split(data, evts, at=0.7)
    n = size(data)[3]
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at * n))
    test_idx = view(idx, (floor(Int, at * n)+1):n)
    data[:, :, train_idx], evts[train_idx, :], data[:, :, test_idx], evts[test_idx, :]
end
