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
    #using CairoMakie
    using Statistics
	using StatsModels
end

# ╔═╡ 74c27e79-a13b-4538-97ff-4dc0670e8237
@revise using DeepRecurrentEncoder

# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = MersenneTwister(1)

# ╔═╡ 549d0f75-be94-4460-9085-022f35613b29
f = @formula 0 ~ 0 + sight+hearing

# ╔═╡ 9ce43061-34bb-4905-b7e2-8fc5f96222cb
f_hearing = @formula 0 ~ 0 + hearing

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100;sfreq=100, sight_effect = 1);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = false

# ╔═╡ d5404ef6-c0c9-45c1-80cd-3a79996889cf
begin
	data_input = Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x
	
dre,ps, st, loss_train = fit(DRE, data_input,f,evts;n_epochs=10,lr=0.1,batch_size=256, hidden_chs = 10)


	
loss_pred,data_pred = DeepRecurrentEncoder.test(dre,data_input,f,evts,ps,st;subset_index=1:10,loss_function=mse)


end

# ╔═╡ a9b24005-ee13-49d5-a208-dace35b68235
# ╠═╡ disabled = true
#=╠═╡
for k in 1:5
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	dre,ps, st, loss_epoch_data, loss_epoch_rsquared_data = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=500,lr=0.1,batch_size=256, hidden_chs = hidden_channels[k])# |> CuArray)
	l,y_pred[k,:,:,:] = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;subset_index=1:10,loss_function = mse)
	push!(lossepochdata, loss_epoch_data)
	push!(lossepochrsquareddata, loss_epoch_rsquared_data)
	push!(loss_test_rsquared,l)
end
  ╠═╡ =#

# ╔═╡ 7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═╡ disabled = true
#=╠═╡
for k in 1:5
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	dre,ps, st, loss_epoch_data, loss_epoch_rsquared_data = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f_hearing,evts;n_epochs=500,lr=0.1,batch_size=256, hidden_chs = hidden_channels[k])# |> CuArray)
	l,y_pred_hearing[k,:,:,:] = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f_hearing,evts,ps,st;subset_index=1:10,loss_function = r_squared)
	push!(lossepochdata_hearing, loss_epoch_data)
	push!(lossepochrsquareddata, loss_epoch_rsquared_data)
	push!(loss_test_rsquared_hearing,l)
end
  ╠═╡ =#

# ╔═╡ 2e46cc71-889a-4bcd-9fee-9ed102298251
# ╠═╡ disabled = true
#=╠═╡
begin
	fig_rsquared = Figure()
	axis_rsquared = Axis(fig_rsquared[1, 1], xticks = (1:5, ["5", "10", "15", "20", "50"]), title = "hidden_channel vs rsquared_error", xlabel = "hidden channels", ylabel = "rsquared_error",)
	lines!(axis_rsquared , 1:length(loss_test_rsquared), loss_test_rsquared, color = :blue, label = "hearing + sight")
	lines!(axis_rsquared , 1:length(loss_test_rsquared_hearing), loss_test_rsquared_hearing, color = :red, label = "hearing")
	lines!(axis_rsquared , 1:length(loss_test_rsquared_hearing), loss_test_rsquared_hearing - loss_test_rsquared, color = :brown, label = "difference")
	legend = axislegend(axis_rsquared)
	fig_rsquared[1, 2] = legend
	fig_rsquared
end
  ╠═╡ =#

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred[5,:, :, 7])', solid_color=:black)

# ╔═╡ 52ba8902-e2ed-4eea-9aa9-66be028a208c
size(y_pred)

# ╔═╡ d869a21d-1ab2-49c2-878b-eb829e8ccb9e
series(data[:,:,15]; solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)

# ╔═╡ 3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
series(mean(y_pred[5,:,:,:], dims=3)[:, :,1]'; solid_color=:black)

# ╔═╡ e1c78107-8fa7-41e8-9045-7e8469ca2d35
size(y_pred)

# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series(data[:,:,6]; solid_color=:black)

# ╔═╡ 16e1e345-343c-4bc3-b6cc-8652285fe063
# ╠═╡ disabled = true
#=╠═╡
begin
	using Plots
end
  ╠═╡ =#

# ╔═╡ f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═╡ disabled = true
#=╠═╡
begin
	Plots.plot!(lossepochdata[1],linecolor=:orange, label="Hidden Channel 5")
	Plots.plot!(lossepochdata[2],linecolor=:brown, label="Hidden Channel 10")
	Plots.plot!(lossepochdata[3], linecolor=:red, label="Hidden Channel 15")
	Plots.plot!(lossepochdata[4], linecolor=:black, label="Hidden Channel 20")
	p1 = Plots.plot!(lossepochdata[5], linecolor=:blue,label="Hidden Channel 50", title="epoch vs loss_mse_epoch", xlabel="epoch", ylabel="loss_epoch")
end
  ╠═╡ =#

# ╔═╡ 4c16266a-f24a-44b8-8773-c3789ef01cd0
begin
	# Flatten the data
	flattened_data_lossmse = []
	line_color_lossmse = [:orange,:brown,:red, :black,:blue]
	labels_lossmse = ["Hidden Channel 5", "Hidden Channel 10", "Hidden Channel 15", "Hidden Channel 20", "Hidden Channel 50"]
	for k in 1:5
		push!(flattened_data_lossmse,[x[1] for x in lossepochdata[k]])
	end
	fig_lossmse = Figure()
	ax_lossmse = Axis(fig_lossmse[1, 1], title = "Loss vs Epoch", xlabel = "Epoch", ylabel = "Loss")
	for (i, data_lossmse) in enumerate(flattened_data_lossmse)
    	lines!(ax_lossmse, 1:length(data_lossmse), data_lossmse, 		 					label=labels_lossmse[i],color=line_color_lossmse[i])
	end
	# Add the legend to the figure
	axislegend(ax_lossmse)
	# Display the figure
	fig_lossmse
end

# ╔═╡ c9a11eac-226a-4d7e-8427-7b7b021138e6
begin
	# Flatten the data
	flattened_data = []
	line_color = [:orange,:brown,:red, :black,:blue]
	labels = ["Hidden Channel 5", "Hidden Channel 10", "Hidden Channel 15", "Hidden Channel 20", "Hidden Channel 50"]
	for k in 1:5
		push!(flattened_data,[x[1] for x in lossepochrsquareddata[k]])
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

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═9ce43061-34bb-4905-b7e2-8fc5f96222cb
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═79712113-2360-4fc9-802d-2e9af5800626
# ╠═d5404ef6-c0c9-45c1-80cd-3a79996889cf
# ╠═a9b24005-ee13-49d5-a208-dace35b68235
# ╠═7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═2e46cc71-889a-4bcd-9fee-9ed102298251
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═52ba8902-e2ed-4eea-9aa9-66be028a208c
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═e1c78107-8fa7-41e8-9045-7e8469ca2d35
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═16e1e345-343c-4bc3-b6cc-8652285fe063
# ╠═f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═4c16266a-f24a-44b8-8773-c3789ef01cd0
# ╠═c9a11eac-226a-4d7e-8427-7b7b021138e6
