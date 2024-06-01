using Random
using Lux
using DeepRecurrentEncoder
using CairoMakie

using Statistics
using LuxCUDA
using Revise
using testdata
includet("../testdata.jl")
# Create an object of the MersenneTwister, use this object to create random number between 0 and 1
rng = MersenneTwister(1)
data, evts = simulate_data(rng, 1000);

begin
    in_chs = size(data, 1) + 1 + 0    # no. of eeg channels + mask + stimuli representation
    hidden_chs = 3
    out_chs = size(data, 1)
    dre = DRE(in_chs, hidden_chs, out_chs)
end

# ╔═╡ be61dbf5-9a71-4f81-bf79-0a066bc8fef1
begin
    ps, st = Lux.setup(rng, dre)
    use_gpu = false
    data_fit = Float32.(data)
    if use_gpu
        ps = ps |> gpu_device()
        st = st |> gpu_device()
        data_fit = CuArray(data_fit)
    end

    ps, st = DeepRecurrentEncoder.(data_fit, dre, ps, st; epochs=50, batch_size=500)
end

begin
    eeg_test = Float32.(permutedims(data[:, :, 1:1], [2, 1, 3]))
    if use_gpu
        eeg_test = CuArray(eeg_test)
    end
    masked_eeg = DeepRecurrentEncoder.add_mask(eeg_test, 0.3)

    l, y_pred, _ = compute_loss(masked_eeg, eeg_test, dre, ps, st)

end


series(Matrix(y_pred[:, :, 1])', solid_color=:black)
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)
