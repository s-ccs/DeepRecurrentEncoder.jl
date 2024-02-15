
function train(eeg, dre, ps, st; lr=0.01, batch_size=32, epochs=1, mask_percentage=0.3)
    @info "Input data is a $(typeof(eeg)) with size $(size(eeg))"

    # conv layer expects input of shape (time, channel, epoch)
    eeg = permutedims(eeg, (2, 1, 3))

    masked_eeg = add_mask(eeg, mask_percentage)
    opt_state = create_optimiser(ps, lr)
    for epoch in 1:epochs
        loss_epoch = 0
        for j in 1:size(eeg, 3)÷batch_size
            start_index = (j - 1) * batch_size + 1
            end_index = j * batch_size
            end_index = end_index > size(eeg, 3) ? size(eeg, 3) : end_index
            eeg_batch = eeg[:, :, start_index:end_index]
            mask_batch = masked_eeg[:, :, start_index:end_index]
            #@show typeof(mask_batch)
            #@show typeof(eeg_batch)
            (loss, y_pred, st), back = pullback(compute_loss, mask_batch, eeg_batch, dre, ps, st)
            loss_epoch += loss
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
        println("-----------------------------------")
        loss_epoch = loss_epoch / size(eeg, 3)
        println("Epoch [$epoch]: Loss " * string(loss_epoch))
    end
    return ps, st
end


# returns average loss
#function test(x, y, dre, ps, st)
#    loss = 0
#    for i in 1:size(x, 3)
#        l, y_pred, st = compute_loss(x, y, dre, ps, st)
#        loss += l
#    end
#    return loss / size(x, 3)
#end




# returns average loss
function test(eeg, dre, ps, st; mask_percentage=0.3, batch_size=32)
    loss = 0
    for j in 1:size(eeg, 3)÷batch_size
        start_index = (j - 1) * batch_size + 1
        end_index = j * batch_size
        end_index = end_index > size(eeg, 3) ? size(eeg, 3) : end_index
        eeg_batch = eeg[:, :, start_index:end_index]
        masked_eeg = add_mask(eeg_batch, mask_percentage)
        #println(size(masked_eeg))
        #println(size(eeg_batch))
        l, y_pred, st = compute_loss(masked_eeg, eeg_batch, dre, ps, st)
        loss += l
    end
    #plot_eeg_prediction(eeg[:, :, 1], dre, ps, st, p)
    return loss / size(eeg, 3)
end