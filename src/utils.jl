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
    n_channels = size(eeg, 2)
    masked[:, 1:n_channels, :] .= eeg
    add_mask!(masked, eeg, p)
end
"""
# returns new array containing eeg with mask or writes the same contents into masked if provided
"""
function add_mask!(masked::AbstractArray{T,3}, eeg::AbstractArray{T,3}, p) where {T}
    index_mask_to = Int(floor(size(eeg, 1) * p))

    # Masking the later part of the input, set the signal to 0 and the mask indicator to 1
    masked[index_mask_to:end, :, :] .= 0
    # is_masked is reversed though
    masked[1:index_mask_to, end, :] .= 0
    masked[index_mask_to:end, end, :] .= 1

    return masked
end


function generate_designmatrix(f::FormulaTerm, evts)
    f = apply_schema(f, schema(f, evts))
    _, X = modelcols(f, evts)
    return X
end


"""
# eeg is a 3d array of shape (channel, time, epoch)
# designmatrix is a Matrix with a row discribing the stimuli for each epoch

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