### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 2ff5fd18-cbf4-11ee-3733-f79124ce7587
begin
    using Pkg
    Pkg.activate("../../")
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

# ╔═╡ 3d6f6750-df4f-441a-94c2-fea9a4d3971f


# ╔═╡ 6be3666a-68c0-4e8b-b04a-9ee5565ed7e1


# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = MersenneTwister(1)

# ╔═╡ 549d0f75-be94-4460-9085-022f35613b29
f = @formula 0 ~ 0 + sight+hearing

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100;sfreq=100);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = false

# ╔═╡ 618d269f-6a3c-4cc4-8b4a-4fb10836122d
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre,ps, st, loss_epoch_data = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256)# |> CuArray)

# ╔═╡ c98522df-7681-4cb6-ad80-47fdf2aeecf5
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;subset_index=1:10)

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred[:, :, 8])', solid_color=:black)

# ╔═╡ 52ba8902-e2ed-4eea-9aa9-66be028a208c
size(data)

# ╔═╡ d869a21d-1ab2-49c2-878b-eb829e8ccb9e
series(data[:,:,15]; solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)

# ╔═╡ 3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
series(mean(y_pred, dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ e1c78107-8fa7-41e8-9045-7e8469ca2d35
size(y_pred)

# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series(data[:,:,6]; solid_color=:black)

# ╔═╡ 2c3aa604-1b30-4962-b211-49a09a59fc51
Plots.plot(loss_epoch_data, label="epoch vs loss_epoch", title="epoch vs loss_epoch", xlabel="epoch", ylabel="loss_epoch")

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═3d6f6750-df4f-441a-94c2-fea9a4d3971f
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═6be3666a-68c0-4e8b-b04a-9ee5565ed7e1
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═618d269f-6a3c-4cc4-8b4a-4fb10836122d
# ╠═79712113-2360-4fc9-802d-2e9af5800626
# ╠═c98522df-7681-4cb6-ad80-47fdf2aeecf5
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═52ba8902-e2ed-4eea-9aa9-66be028a208c
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═e1c78107-8fa7-41e8-9045-7e8469ca2d35
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═16e1e345-343c-4bc3-b6cc-8652285fe063
# ╠═2c3aa604-1b30-4962-b211-49a09a59fc51