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

# ╔═╡ 0459d87b-8adc-4ae2-9254-02338ab58a8d
data, evts = testdata.simulate_data(rng, 100;sfreq=100);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

# ╔═╡ 79712113-2360-4fc9-802d-2e9af5800626
use_gpu = false

# ╔═╡ 6b857be4-57b6-44aa-8de0-c8b16de50dc9
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre_10,ps_10, st_10, loss_epoch_data_10, loss_epoch_rsquared_data_10 = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256, hidden_chs = 10)# |> CuArray)

# ╔═╡ 64dc81db-e3f9-43c3-9acf-3542676430ac
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre_15,ps_15, st_15, loss_epoch_data_15, loss_epoch_rsquared_data_15 = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256, hidden_chs = 15)# |> CuArray)

# ╔═╡ 3be8916b-435e-483f-87a1-4297993dfec1
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l_10,y_pred_10 = DeepRecurrentEncoder.test(dre_10,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps_10,st_10;subset_index=1:10)

# ╔═╡ 99a3a116-3ed4-4e0a-bd24-2380224d446b
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l_15,y_pred_15 = DeepRecurrentEncoder.test(dre_15,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps_15,st_15;subset_index=1:10)

# ╔═╡ 374f5653-589a-4542-b706-c8f120834fb4
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre_20,ps_20, st_20, loss_epoch_data_20, loss_epoch_rsquared_data_20 = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256, hidden_chs = 20)# |> CuArray)

# ╔═╡ 5869f4e0-fd2d-42ea-9e0f-6b287f0e593d
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l_20,y_pred_20 = DeepRecurrentEncoder.test(dre_20,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps_20,st_20;subset_index=1:10)

# ╔═╡ 886ceba2-68c0-4d4c-9812-b2f31c58630f
#dre,ps, st = fit(DRE, Float32.(data))# |> CuArray)
dre_50,ps_50, st_50, loss_epoch_data_50, loss_epoch_rsquared_data_50 = fit(DRE, Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x,f,evts;n_epochs=50,lr=0.1,batch_size=256, hidden_chs = 50)# |> CuArray)

# ╔═╡ e8754f11-4534-462c-b85f-6f32099e52e9
#l,y_pred = DeepRecurrentEncoder.test(dre,(Float32.(data)|> x->use_gpu ? CuArray(x) : x),ps,st;subset_index=1:10)
l_50,y_pred_50 = DeepRecurrentEncoder.test(dre_50,(Float32.(data[:,1:end÷2*2,:])|> x->use_gpu ? CuArray(x) : x),f,evts,ps_50,st_50;subset_index=1:10)

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(Matrix(y_pred_10[:, :, 8])', solid_color=:black)

# ╔═╡ 52ba8902-e2ed-4eea-9aa9-66be028a208c
size(data)

# ╔═╡ d869a21d-1ab2-49c2-878b-eb829e8ccb9e
series(data[:,:,15]; solid_color=:black)

# ╔═╡ 20ad368e-4a3b-43f0-9836-f58adaaa71df
series(mean(data, dims=3)[:, :, 1]; solid_color=:black)

# ╔═╡ 3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
series(mean(y_pred_10, dims=3)[:, :, 1]'; solid_color=:black)

# ╔═╡ e1c78107-8fa7-41e8-9045-7e8469ca2d35
size(y_pred_10)

# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series(data[:,:,6]; solid_color=:black)

# ╔═╡ 53db6390-5e65-4797-b5a4-9023b27b4bbb
hidden_channels = 1:5

# ╔═╡ e71d595d-325e-47dc-80e1-5a9cd583d089
p1 = Plots.plot!(loss_epoch_data_20, linecolor=:blue,label="Hidden Channel 20", title="epoch vs loss_mse_epoch", xlabel="epoch", ylabel="loss_epoch")

# ╔═╡ f98f5c6a-6c98-4f65-9646-7089d7df21c9
Plots.plot!(loss_epoch_data_10, linecolor=:red, label="Hidden Channel 10")

# ╔═╡ 63f94977-e145-4d39-829b-a9329bb37f6e
Plots.plot!(loss_epoch_data_15,linecolor=:brown, label="Hidden Channel 15")

# ╔═╡ a97671a6-f122-4bd9-a11d-e8657102bac8
Plots.plot!(loss_epoch_data_50,linecolor=:yellow, label="Hidden Channel 50")

# ╔═╡ 0eee6f1b-76d8-4617-bdbc-778450a9504c
p2 = Plots.plot(loss_epoch_rsquared_data_50,linecolor=:yellow, label="Hidden Channel 50", title="epoch vs loss_rsquared_epoch", xlabel="epoch", ylabel="loss_epoch")

# ╔═╡ 39576dd1-a41e-45bb-97c8-e171858e6fd4
Plots.plot!(loss_epoch_rsquared_data_10,linecolor=:red, label="Hidden Channel 10")

# ╔═╡ e7203a84-9f80-4c0d-b545-38fb96fa5217
Plots.plot!(loss_epoch_rsquared_data_15,linecolor=:brown, label="Hidden Channel 15")

# ╔═╡ 92f45109-fcef-42cf-a794-7758dcb7a825
Plots.plot!(loss_epoch_rsquared_data_20,linecolor=:blue, label="Hidden Channel 20")

# ╔═╡ Cell order:
# ╠═2ff5fd18-cbf4-11ee-3733-f79124ce7587
# ╠═f2781d64-0017-41a7-a153-a388fef9c027
# ╠═9e4a1e93-9eba-47f5-ba42-785b70e19990
# ╠═74c27e79-a13b-4538-97ff-4dc0670e8237
# ╠═35932abf-10b2-4e85-9090-98dc09499d66
# ╠═549d0f75-be94-4460-9085-022f35613b29
# ╠═0459d87b-8adc-4ae2-9254-02338ab58a8d
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═6b857be4-57b6-44aa-8de0-c8b16de50dc9
# ╠═64dc81db-e3f9-43c3-9acf-3542676430ac
# ╠═3be8916b-435e-483f-87a1-4297993dfec1
# ╠═99a3a116-3ed4-4e0a-bd24-2380224d446b
# ╠═374f5653-589a-4542-b706-c8f120834fb4
# ╠═5869f4e0-fd2d-42ea-9e0f-6b287f0e593d
# ╠═886ceba2-68c0-4d4c-9812-b2f31c58630f
# ╠═e8754f11-4534-462c-b85f-6f32099e52e9
# ╠═79712113-2360-4fc9-802d-2e9af5800626
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═52ba8902-e2ed-4eea-9aa9-66be028a208c
# ╠═d869a21d-1ab2-49c2-878b-eb829e8ccb9e
# ╠═20ad368e-4a3b-43f0-9836-f58adaaa71df
# ╠═3950b729-c5ba-4d12-a39a-2a0f05f5aaa4
# ╠═e1c78107-8fa7-41e8-9045-7e8469ca2d35
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═16e1e345-343c-4bc3-b6cc-8652285fe063
# ╠═53db6390-5e65-4797-b5a4-9023b27b4bbb
# ╠═e71d595d-325e-47dc-80e1-5a9cd583d089
# ╠═f98f5c6a-6c98-4f65-9646-7089d7df21c9
# ╠═63f94977-e145-4d39-829b-a9329bb37f6e
# ╠═a97671a6-f122-4bd9-a11d-e8657102bac8
# ╠═0eee6f1b-76d8-4617-bdbc-778450a9504c
# ╠═39576dd1-a41e-45bb-97c8-e171858e6fd4
# ╠═e7203a84-9f80-4c0d-b545-38fb96fa5217
# ╠═92f45109-fcef-42cf-a794-7758dcb7a825
