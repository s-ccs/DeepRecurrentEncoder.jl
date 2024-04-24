using Lux, Random, Optimisers, Zygote, Plots#, Optimisers, Zygote
using DeepRecurrentEncoder
include("read_data/load_data.jl")

# Seeding
rng = Random.default_rng()
Random.seed!(rng, 1)
Random.TaskLocalRNG()

# Hyperparameters
p = 1.0             # percentage of time at the beginning of each epoch that the input is not masked
lr = 0.01          # learning rate

# Load data
#eeg, times, evts = load_eeg(5,500)

# Model & its state/parameters
in_chs = size(eeg, 1) + 1 + 0    # no. of eeg channels + mask + stimuli representation
hidden_chs = 3
out_chs = size(eeg, 1)
dre = DRE(in_chs, hidden_chs, out_chs)
ps, st = Lux.setup(rng, dre)




# returns average loss
function test(eeg, dre, ps, st, p)
    loss = 0
    batch_size = 32
    for j in 1:size(eeg, 3)÷batch_size
        start_index = (j - 1) * batch_size + 1
        end_index = j * batch_size
        end_index = end_index > size(eeg, 3) ? size(eeg, 3) : end_index
        eeg_batch = replace_missing_with_mean!(eeg[:, :, start_index:end_index])
        masked_eeg = add_mask(eeg_batch, p)
        println(size(masked_eeg))
        println(size(eeg_batch))
        l, y_pred, st = compute_loss(masked_eeg, eeg_batch, dre, ps, st)
        loss += l
    end
    plot_eeg_prediction(eeg[:, :, 1], dre, ps, st, p)
    return loss / size(eeg, 3)
end

function plot_eeg_prediction(eeg_epoch::Array{Union{Missing,Float64},2}, dre, ps, st, p, name="eegPrediction")
    eeg = replace_missing_with_mean!(reshape(eeg_epoch, size(eeg_epoch, 1), size(eeg_epoch, 2), 1))
    masked_eeg = add_mask(eeg, p)
    y_pred, _ = dre(masked_eeg, ps, st)
    Plots.plot([reshape(masked_eeg[1, :, 1], size(masked_eeg, 2)) reshape(y_pred[1, :, 1], size(masked_eeg, 2))], xlabel="time", ylabel="eeg", label=["input" "prediction"], linewidth=2, title="EEG prediction")
    Plots.savefig("plots/" * name * ".png")
end

# ps, st = train(eeg, dre, ps, st, lr, 32, 10)
# eeg = eeg[:,:,1:50]
# loss = test(eeg, dre, ps, st, p)
# println("Average loss on sample of training data: "*string(loss))