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

# ╔═╡ 9ce43061-34bb-4905-b7e2-8fc5f96222cb
f_hearing = @formula 0 ~ 0 + hearing

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100; sfreq=100, sight_effect=1);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = false

# ╔═╡ 28f29ec2-12dd-4c22-871a-3931dca13827
begin
	lossepochdata = []
	lossepochrsquareddata = []
	loss_test_rsquared = []
	hidden_channels = [10,25,50,75,100]
	y_pred = zeros(5, 44, 227, 10)
end

# ╔═╡ a9b24005-ee13-49d5-a208-dace35b68235
for k in 1:5
	#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
	dre,ps, st, loss_epoch_data, loss_epoch_rsquared_data = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=100,lr=0.05,batch_size=256, hidden_chs = hidden_channels[k])# |> CuArray)
	l,y_pred[k,:,:,:] = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;subset_index=1:10,loss_function = mse)
	push!(lossepochdata, loss_epoch_data)
	push!(lossepochrsquareddata, loss_epoch_rsquared_data)
	push!(loss_test_rsquared,l)
end

# ╔═╡ d869a21d-1ab2-49c2-878b-eb829e8ccb9e
series(data[:, :, 5]; solid_color=:black)

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred[5, :, :, 5])', solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)

# ╔═╡ 3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
series(mean(y_pred[1, :, :, :], dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ 51ae940c-6869-42c6-bf0e-d87e4cd530ae
series(mean(y_pred[2, :, :, :], dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ 7443e0dd-e5bc-42c9-8b69-5eb7def05cb1
series(mean(y_pred[3, :, :, :], dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ 6dbdc3ac-dc70-4526-8cd9-5a9cc11e1f91
series(mean(y_pred[4, :, :, :], dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ dc57f80d-9ff4-4b72-a518-6db88966fee7
series(mean(y_pred[5, :, :, :], dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ 4c16266a-f24a-44b8-8773-c3789ef01cd0
begin
    # Flatten the data
    flattened_data_lossmse = []
    line_color_lossmse = [:orange, :brown, :red, :black, :blue]
    labels_lossmse = ["Hidden Channel 5", "Hidden Channel 25", "Hidden Channel 50", "Hidden Channel 75", "Hidden Channel 100"]
    for k in 1:5
        push!(flattened_data_lossmse, [x[1] for x in lossepochdata[k]])
    end
    fig_lossmse = Figure()
    ax_lossmse = Axis(fig_lossmse[1, 1], title="Loss vs Epoch", xlabel="Epoch", ylabel="Loss")
    for (i, data_lossmse) in enumerate(flattened_data_lossmse)
        lines!(ax_lossmse, 1:length(data_lossmse), data_lossmse, label=labels_lossmse[i], color=line_color_lossmse[i])
    end
    # Add the legend to the figure
    axislegend(ax_lossmse)
    # Display the figure
    fig_lossmse
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
# ╠═28f29ec2-12dd-4c22-871a-3931dca13827
# ╠═a9b24005-ee13-49d5-a208-dace35b68235
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═51ae940c-6869-42c6-bf0e-d87e4cd530ae
# ╠═7443e0dd-e5bc-42c9-8b69-5eb7def05cb1
# ╠═6dbdc3ac-dc70-4526-8cd9-5a9cc11e1f91
# ╠═dc57f80d-9ff4-4b72-a518-6db88966fee7
# ╠═4c16266a-f24a-44b8-8773-c3789ef01cd0
