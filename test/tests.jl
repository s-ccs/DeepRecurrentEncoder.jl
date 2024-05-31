data, evts = testdata.simulate_data(rng, 100);
@testset "formula-interface" begin

    X = DeepRecurrentEncoder.generate_designmatrix(@formula(0 ~ 0 + sight), evts)
    @test size(X) == (100, 2)


end

@testset "add_mask" begin


end