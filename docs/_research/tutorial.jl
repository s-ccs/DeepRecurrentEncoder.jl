### A Pluto.jl notebook ###
# v0.20.3

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
f = @formula 0 ~ 0 + sight + hearing

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100;sfreq=100, sight_effect = 1);

# ╔═╡ 3059a2f4-7090-4537-a572-4e3ab561f8dc
use_gpu = false

# ╔═╡ d5404ef6-c0c9-45c1-80cd-3a79996889cf
# SAMPLE TRAINING LOOP
begin
  x_train_data, train_evts, x_test_data, test_evts = train_test_split(data, evts)

  data_input = Float32.(x_train_data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x
  loss = MSELoss()
  dre, ps, st, loss_train, loss_opt = fit(DRE, data_input, f, train_evts; n_epochs=10, lr=0.1, batch_size=256, loss_opt=loss, hidden_chs=25)

  data_test = Float32.(x_test_data[:, 1:end÷2*2, :])

  loss_pred, data_pred = DeepRecurrentEncoder.test(dre, data_test, f, test_evts, ps, st; subset_index=1:10, loss_function=mse)

end


# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series((data_test[:, :, 10]); solid_color=:black)

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(data_pred[:, :, 10]', solid_color=:black)

# ╔═╡ 48541b2a-92c1-4c38-8f79-4b5d67ec50a2
# ╠═╡ disabled = true
#=╠═╡
begin
	loss_epoch_data = []
	hidden_channels = [5, 10, 20,  50,100]
	loss_epoch_mse_data = []
	loss_test_r_squared = []
	y_pred = zeros(Float64, 5, 44, 227, 10)
end;
  ╠═╡ =#

# ╔═╡ 7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═╡ disabled = true
#=╠═╡
for k in 1:5
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
    dre, ps, st, _loss_epoch_data, _loss_epoch_mse_data = fit(DRE, Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x, f, evts; n_epochs=10, lr=0.1, batch_size=256, hidden_chs=hidden_channels[k])# |> CuArray)
	l,y_pred[k,:,:,:] = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;subset_index=1:10,loss_function = r_squared)
    push!(loss_epoch_data, _loss_epoch_data)
    push!(loss_epoch_mse_data, _loss_epoch_mse_data)
    push!(loss_test_r_squared, l)
end
  ╠═╡ =#

# ╔═╡ f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═╡ disabled = true
#=╠═╡
begin
	Plots.plot()
    Plots.plot!(loss_epoch_data[1], linecolor=:orange, label="Hidden Channel 5")
    Plots.plot!(loss_epoch_data[2], linecolor=:brown, label="Hidden Channel 10")
    Plots.plot!(loss_epoch_data[3], linecolor=:red, label="Hidden Channel 20")
    Plots.plot!(loss_epoch_data[4], linecolor=:black, label="Hidden Channel 50")
    p1 = Plots.plot!(loss_epoch_data[5], linecolor=:blue, label="Hidden Channel 100", title="epoch vs loss_mse_epoch", xlabel="epoch", ylabel="loss_epoch")
end
  ╠═╡ =#

# ╔═╡ c9a11eac-226a-4d7e-8427-7b7b021138e6
# ╠═╡ disabled = true
#=╠═╡
begin
	# Flatten the data
	flattened_data = []
	line_color = [:orange,:brown,:red, :black,:blue]
	labels = ["Hidden Channel 5", "Hidden Channel 10", "Hidden Channel 20", "Hidden Channel 50", "Hidden Channel 100"]
	for k in 1:5
        push!(flattened_data, [x[1] for x in loss_epoch_mse_data[k]])
	end
	fig = Figure()
	ax = Axis(fig[1, 1], title = "Loss vs Epoch", xlabel = "Epoch", ylabel = "Loss")
	for (i, data) in enumerate(flattened_data)
    	lines!(ax, 1:length(data), data, label=labels[i], color=line_color[i])
	end

	# Add the legend to the figure
	axislegend(ax)

	# Display the figure
	fig
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═3059a2f4-7090-4537-a572-4e3ab561f8dc
# ╠═d5404ef6-c0c9-45c1-80cd-3a79996889cf
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═48541b2a-92c1-4c38-8f79-4b5d67ec50a2
# ╠═7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═c9a11eac-226a-4d7e-8427-7b7b021138e6
