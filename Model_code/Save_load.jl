using BSON: @load, @save

function name_the_model(Nhidden, seed, Lplan)
    mod_name = 
    "N$Nhidden" *
    "_Lplan$Lplan" *
    "_seed$seed"
end 

function load_model(filename)
    @load filename * "_progress.bson" progress
    @load filename * "_hps.bson" hps
    @load filename * "_rnn.bson" rnn
    @load filename * "_policy_value.bson" π_V
    @load filename * "_prediction.bson" prediction
    return progress, hps, rnn, π_V, prediction
end

function save_model(model, model_properties, optim, filename, progress, βₚ, βₑ, βᵥ, Larena, time_p_action, time_s_action)
    Nhidden = model_properties.Nhidden
    Lplan = model_properties.Lplan
    Nout = model_properties.Nout
    Nin = model_properties. Nin

    rnn = model.rnn
    π_V = model.π_V
    prediction = model.prediction

    hps = Dict(
        "Nhidden" => Nhidden,
        "Larena" => Larena,
        "Nin" => Nin,
        "Nout" => Nout,
        "βₚ" => βₚ,
        "βₑ" => βₑ,
        "βᵥ" => βᵥ,
        "Lplan" => Lplan,
        "time_p_action" => time_p_action,
        "time_s_action" => time_s_action
    )

    @save filename * "_progress.bson" progress
    @save filename * "_optimizer.bson" optim
    @save filename * "_hps.bson" hps
    @save filename * "_rnn.bson" rnn
    @save filename * "_policy_value.bson" π_V
    @save filename * "_prediction.bson" prediction
end

