### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 2ff5fd18-cbf4-11ee-3733-f79124ce7587
begin
    using Pkg
    Pkg.activate("../../docs")
end

# ╔═╡ f2781d64-0017-41a7-a153-a388fef9c027
begin
    using Random
    using PlutoLinks
    using Lux
    using LuxCUDA
    using CairoMakie
    using Statistics
    using StatsModels
    using Plots
    using StableRNGs
end

# ╔═╡ 74c27e79-a13b-4538-97ff-4dc0670e8237
@revise using DeepRecurrentEncoder

# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = StableRNG(1)

# ╔═╡ 549d0f75-be94-4460-9085-022f35613b29
begin
    # Let us define formulas, each with different stimuli
    f_hearing = @formula 0 ~ 0 + hearing
    f_sight = @formula 0 ~ 0 + sight
    f_combined = @formula 0 ~ 0 + sight + hearing
end

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
# Generating data with a random number generator and a customizable sight effect
data, evts = testdata.simulate_data(rng, 100; sfreq=100, sight_effect=1);

# ╔═╡ ad146140-97fa-49b0-a026-966caa14b43d
# Variables to hold values of loss and predictions for input of only sight
begin
    loss_epoch_data_sight = []
    loss_epoch_opt_data_sight = []
    loss_test_opt_sight = []
    y_pred_sight = zeros(Float64, 44, 227, 10)
end

# ╔═╡ d0cc0ebd-5248-4d11-bb1c-17f2cc505953
# Variables to hold values of loss and predictions for input of only hearing
begin
    loss_epoch_data_hearing = []
    loss_epoch_opt_data_hearing = []
    loss_test_opt_hearing = []
    y_pred_hearing = zeros(Float64, 44, 227, 10)
end

# ╔═╡ bb418952-aa97-4edb-ae71-ae8c0cd2844a
# Variables to hold values of loss and predictions for input of combined stimuli
begin
    loss_epoch_data_combined = []
    loss_epoch_opt_data_combined = []
    loss_test_opt_combined = []
    y_pred_combined = zeros(Float64, 44, 227, 10)
end

# ╔═╡ f8ab077f-c976-422c-b348-31163652a2f6
begin
    using DataFrames
    df = DataFrame(
        Stimulus = ["Sight", "Hearing", "Combined"],
        Training_Loss = [loss_epoch_data_sight[1][end], loss_epoch_data_hearing[1][end], loss_epoch_data_combined[1][end]],
        Testing_Loss = [loss_test_opt_sight[1], loss_test_opt_hearing[1], loss_test_opt_combined[1]],
        R²_Score = [mean(loss_epoch_opt_data_sight[1]), mean(loss_epoch_opt_data_hearing[1]), mean(loss_epoch_opt_data_combined[1])]
    )
    df
end

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 3059a2f4-7090-4537-a572-4e3ab561f8dc
# GPU usage
use_gpu = false

# ╔═╡ a9b24005-ee13-49d5-a208-dace35b68235
# Training with stimulus - Sight
begin
    dre_sight, ps_sight, st_sight, loss_epoch_data_recieved_sight, loss_epoch_rsquared_data_sight = fit(DRE, Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x, f_sight, evts; n_epochs=200, lr=0.05, batch_size=256, hidden_chs=75) # |> CuArray)
    l_sight, y_pred_sight[:, :, :] = DeepRecurrentEncoder.test(dre_sight, (Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x), f_sight, evts, ps_sight, st_sight; subset_index=1:10, loss_function=mse)
    push!(loss_epoch_data_sight, loss_epoch_data_recieved_sight)
    push!(loss_epoch_opt_data_sight, loss_epoch_rsquared_data_sight)
    push!(loss_test_opt_sight, l_sight)
end

# ╔═╡ 7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# Training with stimulus - Hearing
begin
    dre_hearing, ps_hearing, st_hearing, loss_epoch_data_recieved_hearing, loss_epoch_rsquared_data_hearing = fit(DRE, Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x, f_hearing, evts; n_epochs=200, lr=0.05, batch_size=256, hidden_chs=75)# |> CuArray)
    l_hearing, y_pred_hearing[:, :, :] = DeepRecurrentEncoder.test(dre_hearing, (Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x), f_hearing, evts, ps_hearing, st_hearing; subset_index=1:10, loss_function=mse)
    push!(loss_epoch_data_hearing, loss_epoch_data_recieved_hearing)
    push!(loss_epoch_opt_data_hearing, loss_epoch_rsquared_data_hearing)
    push!(loss_test_opt_hearing, l_hearing)
end

# ╔═╡ f2fbca49-8916-4b39-a3cf-a330c3919173
# Training with combined stimuli - Sight + Hearing
begin
    dre_combined, ps_combined, st_combined, loss_epoch_data_recieved_combined, loss_epoch_rsquared_data_combined = fit(DRE, Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x, f_combined, evts; n_epochs=200, lr=0.05, batch_size=256, hidden_chs=75)# |> CuArray)
    l_combined, y_pred_combined[:, :, :] = DeepRecurrentEncoder.test(dre_combined, (Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x), f_combined, evts, ps_combined, st_combined; subset_index=1:10, loss_function=mse)
    push!(loss_epoch_data_combined, loss_epoch_data_recieved_combined)
    push!(loss_epoch_opt_data_combined, loss_epoch_rsquared_data_combined)
    push!(loss_test_opt_combined, l_combined)
end

# ╔═╡ ed2a46a3-11c8-4eb0-a681-5029a41eef65
begin
    # Create a figure with a 3x1 grid layout (4 rows, 1 column)
    f = Figure()

    # First subplot for "Original"
    ax1 = Axis(f[1, 1], xlabel="Time", ylabel="Amplitude", title="Original")
    lines!(ax1, data[:, :, 5][:], label="Original", color=:blue)
    axislegend(ax1)

    # Second subplot for "Sight"
    ax2 = Axis(f[2, 1], xlabel="Time", ylabel="Amplitude", title="Sight")
    lines!(ax2, y_pred_sight[:, :, 5][:], label="Hearing", color=:red)
    axislegend(ax2)

    # Third subplot for "Hearing"
    ax3 = Axis(f[3, 1], xlabel="Time", ylabel="Amplitude", title="hearing")
    lines!(ax3, y_pred_hearing[:, :, 5][:], label="hearing", color=:green)
    axislegend(ax3)

	# Third subplot for "Combined"
    ax4 = Axis(f[4, 1], xlabel="Time", ylabel="Amplitude", title="Combined")
    lines!(ax4, y_pred_combined[:, :, 5][:], label="Combined", color=:green)
    axislegend(ax4)

    # Display the figure
    f
end

# ╔═╡ 649545f2-6172-44cb-9eb8-8623ee293be0
begin
	f_ = Figure()

	ax1_ = Axis(f_[1,1], xlabel="Epochs", ylabel="error", title="Error vs epochs")

	lines!(ax1_, loss_epoch_data_sight[1], label="sight", color=:red)

	lines!(ax1_, (loss_epoch_data_hearing[1]), label="hearing", color=:blue)

	lines!(ax1_, (loss_epoch_data_combined[1]), label="combined", color=:black)

	axislegend(ax1_)

	f_
end

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═ad146140-97fa-49b0-a026-966caa14b43d
# ╠═d0cc0ebd-5248-4d11-bb1c-17f2cc505953
# ╠═bb418952-aa97-4edb-ae71-ae8c0cd2844a
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═3059a2f4-7090-4537-a572-4e3ab561f8dc
# ╠═a9b24005-ee13-49d5-a208-dace35b68235
# ╠═7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═f2fbca49-8916-4b39-a3cf-a330c3919173
# ╠═f8ab077f-c976-422c-b348-31163652a2f6
# ╠═ed2a46a3-11c8-4eb0-a681-5029a41eef65
# ╠═649545f2-6172-44cb-9eb8-8623ee293be0
