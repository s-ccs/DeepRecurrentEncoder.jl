using Test
using DeepRecurrentEncoder
using Random
using Lux

include("../docs/testdata.jl");

rng = MersenneTwister(1)
data, evts = simulate_data(rng, 100; sfreq=100, sight_effect=1);
use_gpu = false
@testset "formula-interface" begin

    X = DeepRecurrentEncoder.generate_designmatrix(@formula(0 ~ 0 + sight), evts)
    @test size(X) == (400, 1)

end
@testset "add_mask" begin
    data = rand(Float32, 5, 10, 20)  # Example input
    masked_data = add_mask(data, 0.3)

    @test size(masked_data)[1] == size(data)[1]
    @test size(masked_data)[2] == size(data)[2] + 1
    @test size(masked_data)[3] == size(data)[3]
    @test masked_data != data  # Verify some masking occurred
    @test all(masked_data .>= 0)  # Validate values remain in range
end

@testset "fit and fit!" begin
    data_input = Float32.(data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x
    formula_sight = @formula 0 ~ 0 + sight
    loss = MSELoss()

    dre, ps, st, loss_train, loss_opt = fit(DRE, data_input, formula_sight, evts; n_epochs=10,
        lr=0.1, batch_size=256, loss_opt=loss, hidden_chs=10)

    @test isa(dre, DRE)
    @test length(ps) > 0
    @test size(loss_train) == (10,)
    @test size(loss_opt) == (10,)
end
