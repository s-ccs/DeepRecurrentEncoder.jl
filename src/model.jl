
struct DRE{L,E,D} <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :encoder, :decoder)}
    lstm_cell::L
    encoder::E
    decoder::D
end

"""
# hidden_chs corresponds to the number of filters in the conv layer & the number of hidden units in the LSTM
"""
function DRE(in_chs::Int, hidden_chs::Int, out_chs::Int; kernel_size=4,stride=2)
    return DRE(LSTMCell(hidden_chs => hidden_chs),
        Conv((kernel_size,), (in_chs => hidden_chs), identity, stride=(stride,), use_bias=true, pad=SamePad()),
        ConvTranspose((kernel_size,), (hidden_chs => out_chs), identity, stride=(stride,), use_bias=true, pad=SamePad()))
end

function (s::DRE)(x::AbstractArray{T,3}, ps::NamedTuple,
    st::NamedTuple) where {T}
    # conv layer expects input of shape (time, channel, epoch)
    #x = permutedims(x, (2, 1, 3))
    # apply convolutional layers
 
    encoded, st_encoder = s.encoder(x, ps.encoder, st.encoder)
 
    # apply lstm layers
    x_init, x_rest = Iterators.peel(eachslice(encoded; dims=1))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    #@show y
    ys = reshape(y, 1, size(y, 1), size(y, 2))
    for (i, x_i) in enumerate(x_rest)
        (y, carry), st_lstm = s.lstm_cell((x_i, carry), ps.lstm_cell, st_lstm)
        # @show y
        ys = cat(ys, reshape(y, 1, size(y, 1), size(y, 2)), dims=1)
    end
    # apply deconvolutional layers
    decoded, st_decoder = s.decoder(ys, ps.decoder, st.decoder)
    # reorder dims to match input
    #y = permutedims(decoded, (2, 1, 3))


    st = merge(st, (decoder=st_decoder, lstm_cell=st_lstm, encoder=st_encoder))
    return decoded, st
end

function create_optimiser(ps, lr=0.001)
    opt = Optimisers.ADAM(lr)
    return Optimisers.setup(opt, ps)
end


"""
# eeg is a 3d array of shape (channel, time, epoch)
# stimuli is a DataFrame with a row discribing the stimuli for each epoch
# p is the percentage of the time at the beginning of each epoch that the input is not masked, the masked end of each epoch is 0
# To indicate whether the input is masked, the channel dimension is increased by 1 to add a new binary value for each timestep.
# Each timestep the stimulus is also added to the input, increasing the channel dim by 4 (2 one-hot encoded binary values)
"""
function get_model_input(eeg::Array, stimuli, p)
    masked = Array{Float64,3}(undef, size(eeg, 1) + 1 + 4, size(eeg, 2), size(eeg, 3))
    add_mask!(masked, eeg, p)

    # Add the stimulus to the input
    for epoch in 1:size(eeg, 3)
        s = stimuli.sight[epoch]
        h = stimuli.hearing[epoch]
        idx = (s == "red" ? 0 : 1) + (h == "silent" ? 0 : 2) + 1
        # init all to 0
        masked[size(eeg, 1)+2:size(eeg, 1)+5, :, epoch] .= 0.01
        # set one node to 1
        masked[size(eeg, 1)+1+idx, :, epoch] .= 0.99
    end
    return masked
end