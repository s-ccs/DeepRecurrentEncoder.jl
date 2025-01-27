"""
StatsModels.fit(t::Type{DRE}, args...; kwargs...)

This function provides a shorthand for calling the `StatsModels.fit` method with a default random number generator (`MersenneTwister(1)`).
    
    # Arguments
    - `t::Type{DRE}`: The type of the model to fit.
    - `args...`: Additional positional arguments passed to the fitting function.
    - `kwargs...`: Named arguments passed to the fitting function.
    
    # Returns
    - The result of the fitting process using a default RNG.
    """
StatsModels.fit(t::Type{DRE}, args...; kwargs...) = StatsModels.fit(MersenneTwister(1), t, args...; kwargs...)

function StatsModels.fit(rng, t::Type{DRE}, data::AbstractArray{T}, f::FormulaTerm, events; kwargs...) where {T}
    """
    Fits a DRE model using the given random number generator and design matrix generated from a formula term.

    # Arguments
    - `rng`: The random number generator used for initialization.
    - `t::Type{DRE}`: The type of the model to fit.
    - `data::AbstractArray{T}`: Input data array.
    - `f::FormulaTerm`: The formula defining the design matrix structure.
    - `events`: Events used for design matrix generation.
    - `kwargs...`: Additional keyword arguments for fitting.

    # Returns
    - The result of calling the `fit` method with the generated design matrix.
    """
    designmatrix = T.(generate_designmatrix(f, events))
    @debug typeof(designmatrix), typeof(data)
    fit(rng, t, data, designmatrix; kwargs...)
end

function StatsModels.fit(rng, t::Type{DRE}, data::AbstractArray{T,3}, designmatrix::AbstractArray{T,2}=similar(data, 0, 0); hidden_chs, kwargs...) where {T}
    """
    Fits a DRE model using the provided data and design matrix.

    # Arguments
    - `rng`: The random number generator used for initialization.
    - `t::Type{DRE}`: The type of the model to fit.
    - `data::AbstractArray{T,3}`: Input data array with three dimensions.
    - `designmatrix::AbstractArray{T,2}`: Optional design matrix with two dimensions. Defaults to an empty matrix of similar type to `data`.
    - `hidden_chs`: The number of hidden channels for the DRE model.
    - `kwargs...`: Additional keyword arguments for fitting.

    # Returns
    - `dre`: The fitted DRE model.
    - `ps`: Parameter states of the model.
    - `st`: Training states of the model.
    - `loss_epoch_data`: Loss data per epoch during training.
    - `loss_epoch_opt_data`: Optimized loss data per epoch during training.
    """
    in_chs = size(data, 1) + 1 + size(designmatrix, 2)
    out_chs = size(data, 1)

    dre = DRE(in_chs, hidden_chs, out_chs)
    @debug in_chs, hidden_chs, out_chs
    input_data, output_data = prepare_data(data, designmatrix; kwargs...)
    ps, st, loss_epoch_data, loss_epoch_opt_data = fit!(rng, dre, input_data, output_data; kwargs...)

    return dre, ps, st, loss_epoch_data, loss_epoch_opt_data
end

function StatsModels.fit!(rng, dre::DRE, data_input, data_output; n_epochs=1, batch_size=32, loss_opt=r_squared, kwargs...)
    """
    Trains a DRE model using the provided input and output data.

    # Arguments
    - `rng`: The random number generator used for initialization.
    - `dre::DRE`: The DRE model to train.
    - `data_input`: Input data for the model.
    - `data_output`: Output data for the model.
    - `n_epochs=1`: Number of training epochs. Defaults to 1.
    - `batch_size=32`: Batch size for training. Defaults to 32.
    - `loss_opt=r_squared`: Loss function to optimize. Defaults to `r_squared`.
    - `kwargs...`: Additional keyword arguments for training.

    # Returns
    - `ps`: Parameter states of the trained model.
    - `st`: Training states of the model.
    - `loss_epoch_data`: Loss data per epoch during training.
    - `loss_epoch_opt_data`: Optimized loss data per epoch during training.
    """
    ps, st = Lux.setup(rng, dre)

    if isa(data_output, AbstractGPUArray)
        ps = ps |> gpu_device()
        st = st |> gpu_device()
    end

    ps, st, loss_epoch_data, loss_epoch_opt_data = train(dre, data_input, data_output, ps, st; n_epochs=n_epochs, batch_size=batch_size, loss_opt)
    return ps, st, loss_epoch_data, loss_epoch_opt_data
end

"""
Prepares the input data for training by adding masks and stimuli.

# Arguments
- `data`: Input data array.
- `designmatrix`: The design matrix to integrate into the input data.
- `mask_percentage=0.3`: The percentage of data to mask. Defaults to 0.3.
- `kwargs...`: Additional keyword arguments.

# Returns
- `input_data`: Prepared input data with masks and stimuli added.
- `data`: The original data with dimensions potentially permuted.
"""
function prepare_data(data, designmatrix; mask_percentage=0.3, kwargs...)
    data = permutedims(data, (2, 1, 3))
    masked_data = add_mask(data, mask_percentage)
    stimuli_data = add_stimuli(masked_data, designmatrix)

    if isa(data, LuxCUDA.CuArray)
        input_data = stimuli_data |> LuxCUDA.CuArray
    else
        input_data = stimuli_data
    end

    return input_data, data
end
