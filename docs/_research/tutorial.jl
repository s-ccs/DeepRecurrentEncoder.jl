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
data, evts = testdata.simulate_data(rng, 100; sfreq=100, sight_effect=1);

# ╔═╡ 374e654e-ec60-45f7-9d70-3a4d0eaa168a
evts

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


# ╔═╡ e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
series((data_test[:, :, 10]); solid_color=:black)

# ╔═╡ a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
series(data_pred[:, :, 10]', solid_color=:black)

# ╔═╡ fb6d2eef-2dd4-4f62-b809-4f7102017900
begin
  f_ = Figure()

  ax1_ = Axis(f_[1, 1], xlabel="Epochs", ylabel="error", title="Error vs epochs")

  lines!(ax1_, loss_train, label="loss", color=:red)

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
# ╠═374e654e-ec60-45f7-9d70-3a4d0eaa168a
# ╠═3059a2f4-7090-4537-a572-4e3ab561f8dc
# ╠═d5404ef6-c0c9-45c1-80cd-3a79996889cf
# ╠═e7f78c12-f9b3-46ac-99c2-3be5896cf6cd
# ╠═a0e1c0b6-60a0-4d52-89c2-9f24d88de1b8
# ╠═fb6d2eef-2dd4-4f62-b809-4f7102017900
