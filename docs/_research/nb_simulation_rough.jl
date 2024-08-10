### A Pluto.jl notebook ###
# v0.19.42

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
end

# ╔═╡ 74c27e79-a13b-4538-97ff-4dc0670e8237
@revise using DeepRecurrentEncoder

# ╔═╡ 16e1e345-343c-4bc3-b6cc-8652285fe063
begin
	using Plots
end

# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = MersenneTwister(1)

# ╔═╡ 549d0f75-be94-4460-9085-022f35613b29
f = @formula 0 ~ 0 + sight+hearing

# ╔═╡ 9ce43061-34bb-4905-b7e2-8fc5f96222cb
f_hearing = @formula 0 ~ 0 + hearing

# ╔═╡ e28911d6-3313-4504-90af-0cccae9d0dd8
f_sight = @formula 0 ~ 0 + sight

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100, 1;sfreq=100);

# ╔═╡ 2d073e94-e75f-4c46-950d-7fa9f4db244a
data_0, evts_0 = testdata.simulate_data(rng, 100, 0;sfreq=100);

# ╔═╡ a3dd7def-6c8c-47dc-8280-933e607e424d
data_test_0, evts_test_0 = testdata.simulate_data(rng, 100, 0;sfreq=100);

# ╔═╡ 374e0d33-0ad9-46ab-beb9-51b7e069ff89
data_10, evts_10 = testdata.simulate_data(rng, 100, 10;sfreq=100);

# ╔═╡ 81a879b6-1077-4044-8195-bcc0d6ce1341
data_test_10, evts_test_10 = testdata.simulate_data(rng, 100, 10;sfreq=100);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 915dec40-0685-40b0-9fc0-e99cbb4cf07f
size(data)

# ╔═╡ c56ed7a8-3683-4de9-bb1c-c4d39e569191
let
	a = []
	push!(a,[1,2,3])
	push!(a,[4,5,6])

	a
end

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

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = false

# ╔═╡ a9b24005-ee13-49d5-a208-dace35b68235
begin
		lossepochdata_hearing_0 = []
	lossepochrsquareddata_hearing_0 = []
	loss_test_rsquared_hearing_0 = []
	y_pred_hearing_0 = zeros(Float64, 44, 227, 10)
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	# training the model with 0 effect on sight
	dre,ps, st, lossepochdatahearing_0, lossepochrsquareddatahearing_0 = fit(DRE, Float32.(data_0[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f_hearing,evts_0;n_epochs=100,mask_percentage=0.1,lr=0.1,batch_size=256, hidden_chs = 50)# |> CuArray)
	# testing the model with 0 effect on sight
	l,y_pred_hearing_0[:,:,:] = DeepRecurrentEncoder.test(dre,(Float32.(data_test_0[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f_hearing,evts_test_0,ps,st;subset_index=1:10,mask_percentage=0.1,loss_function = mse)
	push!(lossepochdata_hearing_0, lossepochdatahearing_0)
	push!(lossepochrsquareddata_hearing_0, lossepochrsquareddatahearing_0)
	push!(loss_test_rsquared_hearing_0,l)
end

# ╔═╡ 470169e8-f8e4-4073-b2c0-acfb319c37e6
begin
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	# training the model with 10 effect on sight
		lossepochdata_hearing_10 = []
	lossepochrsquareddata_hearing_10 = []
	loss_test_rsquared_hearing_10 = []
	y_pred_hearing_10 = zeros(Float64, 44, 227, 10)
	
	dre_hearing_10,ps_hearing_10, st_hearing_10, lossepochdatahearing_10, lossepochrsquareddatahearing_10 = fit(DRE, Float32.(data_10[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f_hearing,
		evts_10;n_epochs=100,lr=0.1,mask_percentage = 0.10,batch_size=256, hidden_chs = 50)# |> CuArray)
	# testing the model with 10 effect on sight
	l_hearing_10,y_pred_hearing_10[:,:,:] = DeepRecurrentEncoder.test(dre_hearing_10,(Float32.(data_test_10[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f_hearing,evts_test_10,ps_hearing_10,st_hearing_10;mask_percentage=0.10,subset_index=1:10,loss_function = mse)
	push!(lossepochdata_hearing_10, lossepochdatahearing_10)
	push!(lossepochrsquareddata_hearing_10, lossepochrsquareddatahearing_10)
	push!(loss_test_rsquared_hearing_10,l_hearing_10)
end

# ╔═╡ 7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
begin
		lossepochdata_sight_10 = []
	lossepochrsquareddata_sight_10 = []
	loss_test_rsquared_sight_10 = []
	y_pred_sight_10 = zeros(Float64, 44, 227, 10)

	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	# training the model with 10 effect on sight
	dre_10,ps_10, st_10, loss_epoch_data_10, loss_epoch_rsquared_data_10 = fit(DRE, Float32.(data_10[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts_10;n_epochs=100,lr=0.1,mask_percentage=0.1,batch_size=256, hidden_chs = 50)# |> CuArray)
	# testing the model with 10 effect on sight
	l_10,y_pred_sight_10[:,:,:] = DeepRecurrentEncoder.test(dre_10,(Float32.(data_test_10[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts_test_10,ps_10,st_10;mask_percentage=0.1,subset_index=1:10,loss_function = mse)
	push!(lossepochdata_sight_10, loss_epoch_data_10)
	push!(lossepochrsquareddata_sight_10, loss_epoch_rsquared_data_10)
	push!(loss_test_rsquared_sight_10,l_10)
end

# ╔═╡ bb453f74-bd3e-4ce4-ab68-a902053d44ee
begin
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
		lossepochdata_sight_0 = []
	lossepochrsquareddata_sight_0 = []
	loss_test_rsquared_sight_0 = []
	y_pred_sight_0 = zeros(Float64, 44, 227, 10)
	
	dre_0,ps_0, st_0, lossepochdata_0, lossepochrsquareddata_0 = fit(DRE, Float32.(data_0[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts_0;n_epochs=100,lr=0.1,mask_percentage=0.1,batch_size=256, hidden_chs = 50)# |> CuArray)
	l_0,y_pred_sight_0[:,:,:] = DeepRecurrentEncoder.test(dre_0,(Float32.(data_0[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts_0,ps_0,st_0;mask_percentage=0.1,subset_index=1:10,loss_function = mse)
	push!(lossepochdata_sight_0, lossepochdata_0)
	push!(lossepochrsquareddata_sight_0, lossepochrsquareddata_0)
	push!(loss_test_rsquared_sight_0,l_0)
end

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred_hearing_0[:, :, 7])', solid_color=:black)

# ╔═╡ 52ba8902-e2ed-4eea-9aa9-66be028a208c
size(y_pred_hearing_0)

# ╔═╡ d869a21d-1ab2-49c2-878b-eb829e8ccb9e
series(data[:,:,15]; solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)

# ╔═╡ 864760ee-bf29-436c-8975-6306ca2eb51e
let
	data_plt = data
	evts_plt = evts
	f,ax,h = series(mean(data_plt[:,:,evts_plt.sight.=="blue"], dims=3)[:, :, 1]; solid_color=:blue)
	series!(mean(data_plt[:,:,evts_plt.sight.=="red"], dims=3)[:, :, 1]; solid_color=:red)
f
end

# ╔═╡ dd637f31-99f2-42ff-80d7-92cc40c9abec
let
	_,data_plt = DeepRecurrentEncoder.test(
		dre_0,(Float32.(data_test_0[:,1:end÷2*2,:])|> 
		x->use_gpu ? CuArray(x) : x),
		f,
		evts_test_0,
		ps_0,
		st_0;
		loss_function = mse)
	data_plt = permutedims(data_plt,[2,1,3])
	evts_plt = evts_0
	fig,ax,h = series(mean(data_plt[:,:,evts_plt.sight.=="blue"], dims=3)[:, :, 1]; solid_color=:blue)
	series(fig[1,2],mean(data_plt[:,:,evts_plt.sight.=="red"], dims=3)[:, :, 1]; solid_color=:red)
fig
end

# ╔═╡ bf5d844a-4343-4d16-94d7-2648c5fefab1


# ╔═╡ 79558848-b058-42a9-99ee-e69c202811ad
let
	data_plt = data_0
	evts_plt = evts_0
	f,ax,h = series(mean(data_plt[:,:,evts_plt.sight.=="blue"], dims=3)[:, :, 1]; solid_color=:blue)
	series!(mean(data_plt[:,:,evts_plt.sight.=="red"], dims=3)[:, :, 1]; solid_color=:red)
f
end

# ╔═╡ fb9a12f7-61f3-4b5c-8208-b246ffdfdc81
let
loss_local,data_plt = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;loss_function = mse)
	
data_plt = reshape(data_plt,1:size(data,2)÷2*2,size(data,1),size(data,3))
	
	evts_plt = evts_10
	f2,ax,h = series(mean(data_plt[:,:,evts_plt.sight.=="blue"], dims=3)[:, :, 1]'; solid_color=:blue)
	series!(mean(data_plt[:,:,evts_plt.sight.=="red"], dims=3)[:, :, 1]'; solid_color=:red)
f2
	
end

# ╔═╡ 592d0494-4daa-45d7-bd76-0edcc858acb0
evts

# ╔═╡ 915a4868-6050-4dad-9d9a-797279cb8ade
size(data)

# ╔═╡ 3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
series(mean(y_pred_hearing_10[:,:,:], dims=3)[:, :,1]'; solid_color=:black)

# ╔═╡ e1c78107-8fa7-41e8-9045-7e8469ca2d35
size(y_pred_hearing_10)

# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series(data[:,:,6]; solid_color=:black)

# ╔═╡ f98f5c6a-6c98-4f65-9646-7089d7df21c9
begin
	Plots.plot(log10.(lossepochdata_hearing_10[1]),linecolor=:orange, label="hearing")
	p1 = Plots.plot!(log10.(lossepochdata_sight_10[1]), linecolor=:blue,label="hearing + sight x10", title="epoch vs loss_mse_epoch", xlabel="epoch", ylabel="loss_epoch")
end

# ╔═╡ 622d6551-a806-471d-8287-720a41e4ff19
begin
	Plots.plot(log10.(lossepochdata_hearing_0[1]),linecolor=:orange, label="hearing")
	p2 = Plots.plot!(log10.(lossepochdata_sight_0[1]), linecolor=:blue,label="hearing + sight x0", title="epoch vs loss_mse_epoch", xlabel="epoch", ylabel="loss_epoch")
end

# ╔═╡ f12b6fcb-eb9c-41b3-a547-e247960e4a46
lossepochdata_hearing_sight

# ╔═╡ 4c16266a-f24a-44b8-8773-c3789ef01cd0
begin
	# Flatten the data
	flattened_data_lossmse_1 = []
	flattened_data_lossmse_2 = []
	line_color_lossmse = [:brown,:red]
	labels_lossmse = ["With effect", "without effect"]
	for k in 1:50
		push!(flattened_data_lossmse_1,for x in lossepochdata_hearing_sight)
		push!(flattened_data_lossmse_2,for x in lossepochdata_sight_10)
	end
	fig_lossmse = Figure()
	ax_lossmse = Axis(fig_lossmse[1, 1], title = "Loss vs Epoch", xlabel = "Epoch", ylabel = "Loss")
	lines!(ax_lossmse,1:length(flattened_data_lossmse_1),flattened_data_lossmse_1,label=labels_lossmse[1],color=line_color_lossmse[1])
	lines!(ax_lossmse, 1:length(flattened_data_lossmse_2), flattened_data_lossmse_2, 	label=labels_lossmse[2],color=line_color_lossmse[2])
	# Add the legend to the figure
	axislegend(ax_lossmse)
	# Display the figure
	fig_lossmse
end

# ╔═╡ 7287ec5d-e4a8-4104-9feb-f6e675d987da
flattened_data_lossmse_1

# ╔═╡ c9a11eac-226a-4d7e-8427-7b7b021138e6
# ╠═╡ disabled = true
#=╠═╡
begin
	# Flatten the data
	flattened_data_lossmse = []
	line_color_lossmse = [:brown,:red]
	labels_lossmse = ["With effect", "without effect"]
	push!(flattened_data_lossmse,[x[1] for x in lossepochrsquareddata_hearing_sight])
	push!(flattened_data_lossmse,[x[1] for x in lossepochrsquareddata_sight_10])
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
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═9ce43061-34bb-4905-b7e2-8fc5f96222cb
# ╠═e28911d6-3313-4504-90af-0cccae9d0dd8
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═2d073e94-e75f-4c46-950d-7fa9f4db244a
# ╠═a3dd7def-6c8c-47dc-8280-933e607e424d
# ╠═374e0d33-0ad9-46ab-beb9-51b7e069ff89
# ╠═81a879b6-1077-4044-8195-bcc0d6ce1341
# ╟─374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═915dec40-0685-40b0-9fc0-e99cbb4cf07f
# ╠═c56ed7a8-3683-4de9-bb1c-c4d39e569191
# ╠═a9b24005-ee13-49d5-a208-dace35b68235
# ╠═470169e8-f8e4-4073-b2c0-acfb319c37e6
# ╠═7ea4fc53-ed1e-4442-8ea9-6e294cccecf9
# ╠═bb453f74-bd3e-4ce4-ab68-a902053d44ee
# ╠═2e46cc71-889a-4bcd-9fee-9ed102298251
# ╠═79712113-2360-4fc9-802d-2e9af5800626
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═52ba8902-e2ed-4eea-9aa9-66be028a208c
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═864760ee-bf29-436c-8975-6306ca2eb51e
# ╠═dd637f31-99f2-42ff-80d7-92cc40c9abec
# ╠═bf5d844a-4343-4d16-94d7-2648c5fefab1
# ╠═79558848-b058-42a9-99ee-e69c202811ad
# ╠═fb9a12f7-61f3-4b5c-8208-b246ffdfdc81
# ╠═592d0494-4daa-45d7-bd76-0edcc858acb0
# ╠═915a4868-6050-4dad-9d9a-797279cb8ade
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═e1c78107-8fa7-41e8-9045-7e8469ca2d35
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═16e1e345-343c-4bc3-b6cc-8652285fe063
# ╠═f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═622d6551-a806-471d-8287-720a41e4ff19
# ╠═f12b6fcb-eb9c-41b3-a547-e247960e4a46
# ╠═4c16266a-f24a-44b8-8773-c3789ef01cd0
# ╠═7287ec5d-e4a8-4104-9feb-f6e675d987da
# ╠═c9a11eac-226a-4d7e-8427-7b7b021138e6
