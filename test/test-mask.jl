
rng = StableRNG(1)
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

@testset "training and testing" begin
    x_train_data, train_evts, x_test_data, test_evts = train_test_split(data, evts)

    data_input = Float32.(x_train_data[:, 1:end÷2*2, :]) |> x -> use_gpu ? CuArray(x) : x
    loss = MSELoss()
    dre, ps, st, loss_train, loss_opt = fit(DRE, data_input, f, train_evts; n_epochs=10, lr=0.1, batch_size=256, loss_opt=loss, hidden_chs=10)


    @test isa(dre, DRE)
    @test length(ps) > 0
    @test size(loss_train) == (10,)
    @test size(loss_opt) == (10,)

    data_test = Float32.(x_test_data[:, 1:end÷2*2, :])

    loss_pred, data_pred = DeepRecurrentEncoder.test(dre, data_test, f, test_evts, ps, st; subset_index=1:10, loss_function=mse)

    @test isa(loss_pred, Float32)
    @test size(data_pred) == (size(data_input)[2].size(data_input)[1], size(data_input)[3])
end