struct DRESimple{L,E,D} <: Lux.AbstractExplicitContainerLayer{(:lstm_cell, :decoder, :encoder)}
    lstm_cell::L
    decoder::D
    encoder::E
end

function DRESimple(in_dims::Int, hidden_dims::Int, out_dims::Int)
    return DRESimple(LSTMCell(hidden_dims => hidden_dims), Dense(hidden_dims, out_dims), Dense(in_dims, hidden_dims))
end

function (s::DRESimple)(x::AbstractArray{T,3}, ps::NamedTuple,
    st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))

    encoded, st_encoder = s.encoder(x_init, ps.encoder, st.encoder)
    (y, carry), st_lstm = s.lstm_cell(encoded, ps.lstm_cell, st.lstm_cell)
    y, st_decoder = s.decoder(y, ps.decoder, st.decoder)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    ys = reshape(y, size(y, 1), 1, size(y, 2))

    for (i, x) in enumerate(x_rest)
        encoded, st_encoder = s.encoder(x, ps.encoder, st_encoder)
        (y, carry), st_lstm = s.lstm_cell((encoded, carry), ps.lstm_cell, st_lstm)
        y, st_decoder = s.decoder(y, ps.decoder, st_decoder)
        ys = cat(ys, reshape(y, size(y, 1), 1, size(y, 2)), dims=2)
    end
    st = merge(st, (decoder=st_decoder, lstm_cell=st_lstm, encoder=st_encoder))
    return ys, st
end