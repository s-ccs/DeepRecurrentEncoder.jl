function mse(y_pred, y_true)
    return sum((y_pred .- y_true) .^ 2)
end

function r_squared(y_pred, y_true)
    y_mean = mean(y_true)
    ss_tot = sum((y_true .- y_mean) .^ 2)
    ss_res = sum((y_true .- y_pred) .^ 2)
    return 1 - ss_res / ss_tot
end

function compute_loss(x, y, model, ps, st, loss_function=mse)
    y_pred, st = model(x, ps, st)
    return loss_function(y_pred, y), y_pred, st
end

"""
# returns new array containing eeg with mask or writes the same contents into masked if provided
"""
function add_mask(eeg::AbstractArray, p)
    masked = similar(eeg, size(eeg, 1), size(eeg, 2) + 1, size(eeg, 3))
    add_mask!(masked, eeg, p)
end
"""
# returns new array containing eeg with mask or writes the same contents into masked if provided
"""
function add_mask!(masked::AbstractArray{T,3}, eeg::AbstractArray{T,3}, p) where {T}
    index_mask_to = Int(floor(size(eeg, 1) * p))

    n_channels = size(eeg, 2)
    # Masking the later part of the input, set the signal to 0 and the mask indicator to 1
    masked[index_mask_to:end, 1:n_channels, :] .= 0
    masked[index_mask_to:end, n_channels+1, :] .= 1

    # No masking, copy the signal and set the mask indicator to 0
    masked[1:index_mask_to-1, 1:n_channels, :] .= eeg[1:index_mask_to-1, 1:n_channels, :]
    masked[1:index_mask_to-1, n_channels+1, :] .= 0
    return masked
end
