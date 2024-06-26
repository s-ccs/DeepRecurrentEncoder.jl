
function train(dre::DRE, eeg_in, eeg_out, ps, st; n_epochs=1, lr=0.01, batch_size=32, show_progress=true)

    @info "Input data is a $(typeof(eeg_in)) and $(typeof(eeg_out)) with size $(size(eeg_in)),$(size(eeg_out))"
    # conv layer expects input of shape (time, channel, epoch)
    opt_state = create_optimiser(ps, lr)
    p = Progress(n_epochs)
    
    loss_epoch_rsquared_array = Array{Float64}(undef,n_epochs)
    loss_epoch_array = Array{Float64}(undef,n_epochs)
    for epoch in 1:n_epochs
        loss_epoch = 0
        loss_epoch_rsquared = 0
        for j in range(1, size(eeg_in, 3), step=batch_size)
            start_index = j
            end_index = j + batch_size
            end_index = end_index > size(eeg_in, 3) ? size(eeg_in, 3) : end_index
            eeg_out_batch = eeg_out[:, :, start_index:end_index]
            eeg_in_batch = eeg_in[:, :, start_index:end_index]
            #            @debug size(eeg_in_batch), size(eeg_out_batch), ps, st
            (loss, y_pred, st), back = pullback(compute_loss, eeg_in_batch, eeg_out_batch, dre, ps, st)
            loss_rsquared = r_squared(y_pred, eeg_out_batch)
            loss_epoch += loss
            loss_epoch_rsquared += loss_rsquared
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
        loss_epoch = loss_epoch / size(eeg_in, 3)
        loss_epoch_rsquared = loss_epoch_rsquared / size(eeg_in, 3)
        if show_progress
            next!(p; showvalues=[(:epoch, epoch), (:loss_epoch, loss_epoch), (:loss_epoch_rsquared, loss_epoch_rsquared)])
        end
        loss_epoch_array[epoch] = loss_epoch
        loss_epoch_rsquared_array[epoch] = loss_epoch_rsquared

    end
    return ps, st, loss_epoch_array, loss_epoch_rsquared_array
end



function test(dre, data, ps, st; loss_function, kwargs...)
    @error "not yet implemented"
    test(dre, data, similar(data, 0, 0), ps, st; loss_function, kwargs...)
end

function test(dre, data::AbstractArray{T,3}, f, evts, ps, st;loss_function,kwargs...) where {T}
    designmatrix = T.(generate_designmatrix(f, evts))
    test(dre, data, designmatrix, ps, st; loss_function,kwargs...)
end
function test(dre, data::AbstractArray, designmatrix::AbstractArray, ps, st; loss_function,subset_index=1:size(data, 3), kwargs...)
    input_data, output_data = DeepRecurrentEncoder.prepare_data(data[:, :, subset_index], designmatrix[subset_index, :])
    l, y_pred = test(input_data, output_data, dre, ps, st; loss_function,kwargs...)
    return l, y_pred

end

# returns average loss

function test(eeg_in::AbstractArray{T,3}, eeg_out, dre, ps, st; batch_size=32, loss_function = mse, kwargs...) where {T}
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
    #plot_eeg_prediction(eeg[:, :, 1], dre, ps, st, p)
    return loss / size(eeg_in, 3), y_pred
end

# test(eeg_in...,;loss_function = mse)
# test(eeg_in...,;loss_function = r_squared)
# test(eeg_in...,;loss_function = (args...)->r_rsquared(args...; dim = (1,2)))
# test(eeg_in...,;loss_function = (args...)->r_rsquared(args...; dim = (:time,:channel)))

# test(eeg_in...,;loss_function = r_rsquared; loss_options=(;)) #(;dim = (:time,:channel))))
# loss_function(bla,blub,...; loss_options...)

#DeepRecurrentEncoder.r_squared


            # r_squrared.(eachrow(y_pred),eachrow(eeg_out_batch)) # mapslices
