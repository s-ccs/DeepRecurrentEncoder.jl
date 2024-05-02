using UnfoldSim
using Random
#using PyMNE
using CSV
# returns tuple (matrix of shape 7*basisLength*4000, df with 4000 rows and 2 columns stating the values of the observation)) 
<<<<<<< HEAD
function simulate_data(rng, epochs;noiselevel=1)
=======
function simulate_data(rng, epochs; sfreq=100)
>>>>>>> 71d217a931eaf813b0af81d1cafd7434fcb7f2c3
    # start by defining the design / event-table
    design = SingleSubjectDesign(;
        conditions=Dict(:sight => ["red", "blue"],
            :hearing => ["silent", "loud"])) |> d -> RepeatDesign(d, epochs)

    # next define a ground-truth signal + relation to events/design with Wilkinson Formulas
    p1 = LinearModelComponent(;
        basis=p100(sfreq=sfreq),
        formula=@formula(0 ~ 1),
        β=[5]
    )

    n1 = LinearModelComponent(;
        basis=n170(sfreq=sfreq),
        formula=@formula(0 ~ 1 + sight),
        β=[5, -3]
    )

    p3 = LinearModelComponent(;
        basis=p300(sfreq=sfreq),
        formula=@formula(0 ~ 1 + sight),
        β=[-5, 1]
    )
    hart = headmodel(type="hartmut")
    #mc = UnfoldSim.MultichannelComponent(c, hart => "Left Postcentral Gyrus")
    mc = UnfoldSim.MultichannelComponent(p1, hart => hart.cortical["label"][50])
    mc2 = UnfoldSim.MultichannelComponent(n1, hart => hart.cortical["label"][100])
    mc3 = UnfoldSim.MultichannelComponent(p3, hart => hart.cortical["label"][150])
    components = [mc, mc2, mc3]

    # finally, define some Onset Distribution and Noise, and simulate!
    # channel, time, epoch
<<<<<<< HEAD
    data = simulate(rng, design, components, UniformOnset(; offset=150, width=4), PinkNoise(;noiselevel=noiselevel), return_epoched=true)
=======
    data = simulate(rng, design, components, UniformOnset(; offset=Int(round(1.5 * sfreq)), width=Int(round(0.04 * sfreq))), PinkNoise(), return_epoched=true)
>>>>>>> 71d217a931eaf813b0af81d1cafd7434fcb7f2c3
    return (data[1], data[2])
end


function load_eeg(nChannels::Int, nEpochs::Int, eeg_path="data/sub-34_task-WLFO_eeg.set", event_path="data/sub-34_task-WLFO_events.tsv")
    # 128ch channels
    eeglabdata = PyMNE.io.read_raw_eeglab(eeg_path)
    eeglabdata.resample(256)
    events = CSV.read(event_path, DataFrame)
    events.latency = events.onset ./ pyconvert(Float64, eeglabdata.info["sfreq"])
    datamatrix = pyconvert(Array, eeglabdata.get_data(units="uV"))
    evts_fixationonly = subset(events, :type => x -> x .== "fixation")
    data_e, times = Unfold.epoch(data=datamatrix, tbl=evts_fixationonly, τ=(-0.3, 1.0), sfreq=pyconvert(Float64, eeglabdata.info["sfreq"]))
    return data_e[1:nChannels, :, 1:nEpochs], times, evts_fixationonly[1:nEpochs, :]
end