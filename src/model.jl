"""
DRE{L,E,D}

Defines a Deep Recurrent Encoder (DRE) structure with LSTM, encoder, and decoder components.

# Fields
- `lstm_cell::L`: The LSTM cell component of the DRE.
- `encoder::E`: The encoder component of the DRE.
- `decoder::D`: The decoder component of the DRE.
"""
struct DRE{L,E,D} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :encoder, :decoder)}
    lstm_cell::L
    encoder::E
    decoder::D
end

"""
Constructs a DRE model with specified input, hidden, and output channels.
Current implementation based on publication : arXiv:2103.02339

# Arguments
- `in_chs::Int`: Number of input channels.
- `hidden_chs::Int`: Number of hidden channels (filters in convolution and units in LSTM).
- `out_chs::Int`: Number of output channels.
- `kernel_size=4`: Kernel size for convolutional layers. Defaults to 4.
- `stride=2`: Stride for convolutional layers. Defaults to 2.

# Returns
- A new `DRE` model instance.
"""
function DRE(in_chs::Int, hidden_chs::Int, out_chs::Int; kernel_size=4, stride=2)
    return DRE(LSTMCell(hidden_chs => hidden_chs),
        Conv((kernel_size,), (in_chs => hidden_chs), identity, stride=(stride,), use_bias=true, pad=SamePad()),
        ConvTranspose((kernel_size,), (hidden_chs => out_chs), identity, stride=(stride,), use_bias=true, pad=SamePad()))
end

"""
Applies the DRE model to the input data.

# Arguments
- `s::DRE`: The DRE model instance.
- `x::AbstractArray{T,3}`: Input data with three dimensions (time, channel, epoch).
- `ps::NamedTuple`: Parameter states of the model.
- `st::NamedTuple`: Training states of the model.

# Returns
- `decoded`: The decoded output after applying the model.
- `st`: Updated states of the model components.
"""
function (s::DRE)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    encoded, st_encoder = s.encoder(x, ps.encoder, st.encoder)

    x_init, x_rest = Iterators.peel(eachslice(encoded; dims=1))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    ys = reshape(y, 1, size(y, 1), size(y, 2))

    for (i, x_i) in enumerate(x_rest)
        (y, carry), st_lstm = s.lstm_cell((x_i, carry), ps.lstm_cell, st_lstm)
        ys = cat(ys, reshape(y, 1, size(y, 1), size(y, 2)), dims=1)
    end

    decoded, st_decoder = s.decoder(ys, ps.decoder, st.decoder)
    st = merge(st, (decoder=st_decoder, lstm_cell=st_lstm, encoder=st_encoder))
    return decoded, st
end

"""
Creates an optimizer for training the model.

# Arguments
- `ps`: Model parameters to optimize.
- `lr=0.001`: Learning rate for the optimizer. Defaults to 0.001.

# Returns
- A setup optimizer instance using the ADAM algorithm.
"""
function create_optimiser(ps, lr=0.001)
    opt = Optimisers.ADAM(lr)
    return Optimisers.setup(opt, ps)
end
