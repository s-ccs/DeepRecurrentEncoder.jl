using Lux, Optimisers, Zygote, Random, Plots
using DeepRecurrentEncoder

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 1)
Random.TaskLocalRNG()

function train(x, y, dre, ps, st, epochs=2)
    opt_state = create_optimiser(ps)
    #plot_epoch(x, y, 1, dre, ps, st)
    for epoch in 1:epochs
        loss = undef
        for j in 1:size(x, 3)÷32
            batch_x = x[:, :, j:j+31]
            batch_y = y[:, :, j:j+31]
            (loss, y_pred, st), back = pullback(compute_loss, batch_x, batch_y, dre, ps, st)
            gs = back((one(loss), nothing, nothing))[4]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end
    end
    #plot_epoch(x, y, 1, dre, ps, st)
    return ps, st
end





# calculate feature importance of stimuli by shuffling one of them at a time and computing the difference of the loss to the original
# return map of stimuli to loss difference
function variable_importance_analysis(y, df, dre, ps, st, p)
    x = get_model_input(y, df, p)
    loss_original = test(x, y, dre, ps, st)

    df_sight = copy(df)
    shuffle!(df_sight.sight)
    x_sight = get_model_input(y, df_sight, p)
    loss_sight = test(x_sight, y, dre, ps, st)

    df_hearing = copy(df)
    shuffle!(df_hearing.hearing)
    x_hearing = get_model_input(y, df_hearing, p)
    loss_hearing = test(x_hearing, y, dre, ps, st)

    return Dict("sight" => loss_sight - loss_original, "hearing" => loss_hearing - loss_original)
end

function plot_epoch(x, label, epoch, dre, ps, st)
    xi = x[:, :, epoch]
    xi = reshape(xi, size(xi, 1), size(xi, 2), 1)
    y_pred, _ = dre(xi, ps, st)
    Plots.plot([reshape(xi[1, :, :], size(xi, 2)) reshape(label[1, :, epoch], size(xi, 2)) reshape(y_pred[1, :, 1], size(xi, 2))], xlabel="time", ylabel="eeg", label=["input" "label" "prediction"], linewidth=2, title="Epoch " * string(epoch))
    Plots.savefig("plots/epoch_" * string(epoch) * ".png")
end




function train_test(model, text, train_data_size, test_data_size, epochs)
    (y, df) = simulate_data(rng, train_data_size)

    x = get_model_input(y, df, p)
    println(text * "Input shape: " * string(size(x)))
    println(text * "in, hidden & out channels: " * string((in_chs, hidden_chs, out_chs)))

    ps, st = Lux.setup(rng, model)
    ps, st = train(x, y, model, ps, st, epochs)

    println(text * "Average loss on training data: " * string(test(x, y, model, ps, st)))

    (y, df) = simulate_data(rng, test_data_size)
    x = get_model_input(y, df, p)
    println(text * "Average loss on test data: " * string(test(x, y, model, ps, st)))


    importance_dict = variable_importance_analysis(y, df, model, ps, st, p)
    println(text * "Variable importance: " * string(importance_dict))
end

p = 0.3
in_chs = 12
hidden_chs = 25
out_chs = 7

dre = DRE(in_chs, hidden_chs, out_chs)
dre_simple = DRESimple(in_chs, hidden_chs, out_chs)

train_test(dre_simple, "simple |>   ", 4000, 100, 2)
train_test(dre, "with conv |>   ", 4000, 100, 2)

