using UnfoldSim
using Random
#using PyMNE
using CSV
# returns tuple (matrix of shape 7*basisLength*4000, df with 4000 rows and 2 columns stating the values of the observation)) 
function simulate_data(rng, epochs; sight_effect=1, sfreq=100)
    # SingleSubjectDesign is used to create a sample data set for experimantal purpose with several cominations of the variable values
    # sight and hearing 
    design = SingleSubjectDesign(;
        conditions=Dict(:sight => ["red", "blue"],
            :hearing => ["silent", "loud"])) |> d -> RepeatDesign(d, epochs)

    #Event-related potentials (ERPs) are very small voltages generated in the brain structures in response to specific events or stimuli
    #Here we are interested in 3 ERP points positive peak after 100ms which is the initial , Hanning window
    # next define a ground-truth signal + relation to events/design with Wilkinson Formulas
    p1 = LinearModelComponent(;
        basis=p100(sfreq=sfreq),
        formula=@formula(0 ~ 1),
        β=[5]
    )

    # Subjects react to the face stimulus which is found to be at negative peak after 170ms 
    n1 = LinearModelComponent(;
        basis=n170(sfreq=sfreq),
        formula=@formula(0 ~ 1 + sight),
        β=[5, -3 * sight_effect]
    )

    #Subject detects the target at positive deflection after 300ms
    p3 = LinearModelComponent(;
        basis=p300(sfreq=sfreq),
        formula=@formula(0 ~ 1 + sight),
        β=[-5, 1 * sight_effect]
    )
    hart = headmodel(type="hartmut")
    #6000 part around the brain we have selected 3 parts here
    #mc = UnfoldSim.MultichannelComponent(c, hart => "Left Postcentral Gyrus")
    mc = UnfoldSim.MultichannelComponent(p1, hart => hart.cortical["label"][50])
    mc2 = UnfoldSim.MultichannelComponent(n1, hart => hart.cortical["label"][100])
    mc3 = UnfoldSim.MultichannelComponent(p3, hart => hart.cortical["label"][150])
    components = [mc, mc2, mc3]

    # finally, define some Onset Distribution and Noise, and simulate!
    # channel, time, epoch
    data = simulate(rng, design, components, UniformOnset(; offset=Int(round(1.5 * sfreq)), width=Int(round(0.04 * sfreq))), PinkNoise(), return_epoched=true)
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