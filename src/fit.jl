
StatsModels.fit(t::Type{DRE}, args...; kwargs...) = StatsModels.fit(MersenneTwister(1), t, args...; kwargs...)


function StatsModels.fit(rng, t::Type{DRE}, data::AbstractArray{T}, f::FormulaTerm, events; kwargs...) where {T}

    designmatrix = T.(generate_designmatrix(f, events))


    #designmatrix = T(eltype(data).(X)
    @debug typeof(designmatrix), typeof(data)
    fit(rng, t, data, designmatrix; kwargs...)
end

function StatsModels.fit(rng, t::Type{DRE}, data::AbstractArray{T,3}, designmatrix::AbstractArray{T,2}=similar(data, 0, 0); hidden_chs, kwargs...) where {T}
    in_chs = size(data, 1) + 1 + size(designmatrix, 2) # channel + mask + stimuli
    out_chs = size(data, 1)

    dre = DRE(in_chs, hidden_chs, out_chs)
    @debug in_chs, hidden_chs, out_chs
    input_data, output_data = prepare_data(data, designmatrix; kwargs...)
    # permuting dimensions


    ps, st, loss_epoch_data, loss_epoch_rsquared_data = fit!(rng, dre, input_data, output_data; kwargs...)

    return dre, ps, st, loss_epoch_data, loss_epoch_rsquared_data


end

"""
Todo: Can be greatly improved by generating the stimulus matrix once and then masking + add_stimuli :shrug:
"""
function prepare_data(data, designmatrix; mask_percentage=0.3, kwargs...)

    data = permutedims(data, (2, 1, 3))

    masked_data = add_mask(data, mask_percentage)
    stimuli_data = add_stimuli(masked_data, designmatrix)
    if isa(data, LuxCUDA.CuArray)
        #masked_data_cpugpu = masked_data |> LuxCUDA.CuArray
        input_data = stimuli_data |> LuxCUDA.CuArray
    else
        #masked_data_cpugpu = masked_data
        input_data = stimuli_data
    end
    return input_data, data

end
function StatsModels.fit!(rng, dre::DRE, data_input, data_output; n_epochs=1, batch_size=32, kwargs...)

    ps, st = Lux.setup(rng, dre)
    if isa(data_output, LuxCUDA.CuArray)
        ps = ps |> gpu_device()
        st = st |> gpu_device()
    end

    ps, st, loss_epoch_data, loss_epoch_rsquared_data = train(dre, data_input, data_output, ps, st; n_epochs=n_epochs, batch_size=batch_size)
    return ps, st, loss_epoch_data, loss_epoch_rsquared_data
end