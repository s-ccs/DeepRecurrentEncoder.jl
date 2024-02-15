### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# ╔═╡ 2ff5fd18-cbf4-11ee-3733-f79124ce7587
begin
	using Pkg
	Pkg.activate("../../")
end

# ╔═╡ 96ba849e-7662-4178-80cb-d37707c9915e
using Random

# ╔═╡ 245689c3-2be6-44e6-a603-2c8b00d11fad
using PlutoLinks

# ╔═╡ 7c9ceafd-23f0-4730-a000-5fff7d563d5a
using Lux

# ╔═╡ 576b8ed6-58da-4c6d-a447-8aab6094241a
@revise using DeepRecurrentEncoder

# ╔═╡ 09d4c71d-aea7-4baf-bc18-200c0824f27b
using CairoMakie


# ╔═╡ fc1337b8-19aa-4006-b731-e6e851780a6e
using Statistics

# ╔═╡ 13bc5bd4-a025-44de-b89a-5ba1481d1c3e
using LuxCUDA

# ╔═╡ 9e4a1e93-9eba-47f5-ba42-785b70e19990
testdata = @ingredients("../../testdata.jl")

# ╔═╡ 35932abf-10b2-4e85-9090-98dc09499d66
rng = MersenneTwister(1)

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data,evts = testdata.simulate_data(rng,1000);

# ╔═╡ c7e35d81-b1e9-4ff6-b22c-b5f4bec1cedb
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
	 	ps = ps|>gpu_device()
	 	st = st|>gpu_device()
	 	data_fit = CuArray(data_fit)
	 end
	
    ps, st = DeepRecurrentEncoder.train(data_fit, dre, ps, st; epochs=50,batch_size=500)
 end


# ╔═╡ 9d58c85e-9ee7-47d6-b608-2b527762a00e


# ╔═╡ 77daa494-7a32-407e-a367-691aad282ed5
size(data)

# ╔═╡ c025888e-089a-42b4-981b-ee59de85327a
begin
	eeg_test = Float32.(permutedims(data[:,:,1:1],[2,1,3]))
	if use_gpu
	eeg_test = CuArray(eeg_test)
	end
	masked_eeg = DeepRecurrentEncoder.add_mask(eeg_test, 0.3)

l, y_pred, _ = compute_loss(masked_eeg,eeg_test, dre, ps, st)

end


# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred[:,:,1])',solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data,dims=3)[:,:,1];solid_color=:black)

# ╔═╡ 37401bba-6489-4f8c-9abb-2a286ebceb5d
size(data)

# ╔═╡ 5f78fdd2-1450-4559-9d53-5cab318bb9af


# ╔═╡ 3f69ba54-dcbb-4794-8084-154e0f11703f
methods(DeepRecurrentEncoder.add_mask)


# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═96ba849e-7662-4178-80cb-d37707c9915e
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═245689c3-2be6-44e6-a603-2c8b00d11fad
# ╠═7c9ceafd-23f0-4730-a000-5fff7d563d5a
# ╠═576b8ed6-58da-4c6d-a447-8aab6094241a
# ╠═c7e35d81-b1e9-4ff6-b22c-b5f4bec1cedb
# ╠═be61dbf5-9a71-4f81-bf79-0a066bc8fef1
# ╠═9d58c85e-9ee7-47d6-b608-2b527762a00e
# ╠═77daa494-7a32-407e-a367-691aad282ed5
# ╠═c025888e-089a-42b4-981b-ee59de85327a
# ╠═09d4c71d-aea7-4baf-bc18-200c0824f27b
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═fc1337b8-19aa-4006-b731-e6e851780a6e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═37401bba-6489-4f8c-9abb-2a286ebceb5d
# ╠═13bc5bd4-a025-44de-b89a-5ba1481d1c3e
# ╠═5f78fdd2-1450-4559-9d53-5cab318bb9af
# ╠═3f69ba54-dcbb-4794-8084-154e0f11703f
