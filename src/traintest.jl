function train(dre::DRE, eeg_in, eeg_out, ps, st; n_epochs=1, lr=0.01, batch_size=32, loss_opt, show_progress=true)
    """
    Train the Deep Recurrent Encoder (DRE) model.

    # Arguments
    - `dre::DRE`: The Deep Recurrent Encoder model.
    - `eeg_in`: Input EEG data, typically a 3D array (time, channel, epoch).
    - `eeg_out`: Target EEG data, same shape as `eeg_in`.
    - `ps`: Model parameters.
    - `st`: Initial state of the model.
    - `n_epochs::Int=1`: Number of training epochs.
    - `lr::Float64=0.01`: Learning rate for the optimizer.
    - `batch_size::Int=32`: Size of the batches for training.
    - `loss_opt`: Loss function for optimization.
    - `show_progress::Bool=true`: Display progress during training.

    # Returns
    A tuple `(ps, st, loss_epoch_array, loss_epoch_opt_array)`:
    - `ps`: Updated model parameters.
    - `st`: Updated state of the model.
    - `loss_epoch_array`: Array of average losses per epoch.
    - `loss_epoch_opt_array`: Array of average optimized losses per epoch.

    """
    @info "Input data is a $(typeof(eeg_in)) and $(typeof(eeg_out)) with size $(size(eeg_in)),$(size(eeg_out))"
    opt_state = create_optimiser(ps, lr)
    loss_epoch_opt_array = Array{Float64}(undef, n_epochs)
    loss_epoch_array = Array{Float64}(undef, n_epochs)

    @progress name = "training progress" threshold = 0.005 for epoch in 1:n_epochs
        loss_epoch = 0
        loss_epoch_opt = 0
        for j in range(1, size(eeg_in, 3), step=batch_size)
            start_index = j
            end_index = j + batch_size
            end_index = end_index > size(eeg_in, 3) ? size(eeg_in, 3) : end_index
            eeg_out_batch = eeg_out[:, :, start_index:end_index]
            eeg_in_batch = eeg_in[:, :, start_index:end_index]
            @debug size(eeg_in_batch), size(eeg_out_batch)
            (loss, y_pred, st), back = pullback(compute_loss, eeg_in_batch, eeg_out_batch, dre, ps, st)
            loss_opt_values = loss_opt(y_pred, eeg_out_batch)
            loss_epoch += loss
            loss_epoch_opt += loss_opt_values
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
        loss_epoch = loss_epoch / size(eeg_in, 3)
        loss_epoch_opt = loss_epoch_opt / size(eeg_in, 3)

        loss_epoch_array[epoch] = loss_epoch
        loss_epoch_opt_array[epoch] = loss_epoch_opt
    end
    return ps, st, loss_epoch_array, loss_epoch_opt_array
end

function test(dre, data, ps, st; kwargs...)
    """
    Placeholder function for testing the model.

    # Arguments
    - `dre`: The Deep Recurrent Encoder model.
    - `data`: Input data for testing.
    - `ps`: Model parameters.
    - `st`: Model state.
    - `kwargs...`: Additional keyword arguments.

    # Returns
    Raises an error since this method is not implemented.
    """
    @error "not yet implemented"
    test(dre, data, similar(data, 0, 0), ps, st; kwargs...)
end

function test(dre, data::AbstractArray{T,3}, f, evts, ps, st; kwargs...) where {T}
    """
    Test the model using a design matrix generated from events.

    # Arguments
    - `dre`: The Deep Recurrent Encoder model.
    - `data::AbstractArray{T,3}`: Input data for testing.
    - `f`: Function to generate the design matrix.
    - `evts`: Event data for design matrix generation.
    - `ps`: Model parameters.
    - `st`: Model state.
    - `kwargs...`: Additional keyword arguments.

    # Returns
    Calls the next `test` function with the generated design matrix.
    """
    designmatrix = T.(generate_designmatrix(f, evts))
    test(dre, data, designmatrix, ps, st; kwargs...)
end

function test(dre, data::AbstractArray, designmatrix::AbstractArray, ps, st; subset_index=1:size(data, 3), kwargs...)
    """
    Test the model with specified data and design matrix.

    # Arguments
    - `dre`: The Deep Recurrent Encoder model.
    - `data::AbstractArray`: Input data for testing.
    - `designmatrix::AbstractArray`: Design matrix for testing.
    - `ps`: Model parameters.
    - `st`: Model state.
    - `subset_index`: Indices of the data to be tested.
    - `kwargs...`: Additional keyword arguments.

    # Returns
    A tuple `(l, y_pred)`:
    - `l`: Loss value.
    - `y_pred`: Predicted outputs.
    """
    input_data, output_data = DeepRecurrentEncoder.prepare_data(data[:, :, subset_index], designmatrix[subset_index, :])
    l, y_pred = test(input_data, output_data, dre, ps, st; kwargs...)
    return l, y_pred
end

function test(eeg_in::AbstractArray{T,3}, eeg_out, dre, ps, st; batch_size=32, loss_function=mse, kwargs...) where {T}
    """
    Test the model on batched EEG data.

    # Arguments
    - `eeg_in::AbstractArray{T,3}`: Input EEG data (time, channel, epoch).
    - `eeg_out`: Target EEG data.
    - `dre`: The Deep Recurrent Encoder model.
    - `ps`: Model parameters.
    - `st`: Model state.
    - `batch_size::Int=32`: Size of the batches for testing.
    - `loss_function`: Function to compute the loss.
    - `kwargs...`: Additional keyword arguments.

    # Returns
    A tuple `(loss, y_pred)`:
    - `loss`: Average loss over all batches.
    - `y_pred`: Predicted outputs.
    """
    @debug size(eeg_in), size(eeg_out)
    loss = 0
    y_pred = Array{T}(undef, size(eeg_out)...)
    for j in range(1, size(eeg_in, 3), step=batch_size)
        start_index = j
        end_index = j + batch_size
        end_index = end_index > size(eeg_in, 3) ? size(eeg_in, 3) : end_index
        eeg_in_batch = eeg_in[:, :, start_index:end_index]
        eeg_out_batch = eeg_out[:, :, start_index:end_index]
        @debug typeof(eeg_in_batch), typeof(eeg_out_batch), typeof(y_pred)
        l, y_pred_tmp, st = compute_loss(eeg_in_batch, eeg_out_batch, dre, ps, st, loss_function)
        y_pred[:, :, start_index:end_index] .= Array(y_pred_tmp)
        loss += l
    end
    return loss / size(eeg_in, 3), y_pred
end
