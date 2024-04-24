### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 2ff5fd18-cbf4-11ee-3733-f79124ce7587
begin
    using Pkg
    Pkg.activate("../../")
end

# ╔═╡ 96ba849e-7662-4178-80cb-d37707c9915e
begin
    using Random
    using PlutoLinks
    using Lux
    using LuxCUDA
    using CairoMakie
    using Statistics
	using StatsModels
end

# ╔═╡ 576b8ed6-58da-4c6d-a447-8aab6094241a
@revise using DeepRecurrentEncoder

# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = MersenneTwister(1)

# ╔═╡ 549d0f75-be94-4460-9085-022f35613b29
f = @formula 0 ~ 0 + sight+hearing

# ╔═╡ 8490ed9d-dfe4-4887-b9dc-ccf79e8b04dd


# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100;sfreq=100);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = true

# ╔═╡ 618d269f-6a3c-4cc4-8b4a-4fb10836122d
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre,ps, st = fit(DRE, Float32.(data)|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256)# |> CuArray)

# ╔═╡ 9ccac6e4-5526-4b0c-851e-f1b0b698a40b
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),f,evts,ps,st;subset_index=1:10)


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

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═96ba849e-7662-4178-80cb-d37707c9915e
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═8490ed9d-dfe4-4887-b9dc-ccf79e8b04dd
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═576b8ed6-58da-4c6d-a447-8aab6094241a
# ╠═618d269f-6a3c-4cc4-8b4a-4fb10836122d
# ╠═79712113-2360-4fc9-802d-2e9af5800626
# ╠═9ccac6e4-5526-4b0c-851e-f1b0b698a40b
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═52ba8902-e2ed-4eea-9aa9-66be028a208c
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═e1c78107-8fa7-41e8-9045-7e8469ca2d35
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
