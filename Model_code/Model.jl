struct ModularModel
    rnn::Any
    π_V::Any
    prediction::Any
end

struct ModelProperties
    Nhidden::Int
    Lplan::Int
    Nin::Int
    Nplan_in::Int
    Nout::Int
    Npred_out::Int
    greedy_actions::Bool
    no_planning::Any #if true, never do planning 
end

function ModelProperties(; Nhidden, Lplan, Nin, Nplan_in, Nout, Npred_out, greedy_actions, no_planning = false)
    return ModelProperties(Nhidden, Lplan, Nin, Nplan_in, Nout, Npred_out, greedy_actions, no_planning)
end

function build_model(Nin, Nhidden, Npred_out, Naction)
"""
Build the model with 3 modules
"""
    # module 1 - recurrent part (generates hidden state)
    rnn = Chain(GRU(Nin, Nhidden))

    # module 2 - takes in hidden state and generates policy and value funtion
    π_V = Chain(Dense(Nhidden, Naction + 1)) # Output dimension is 6: policy (5 actions) + 1 value function

    # module 3 - internal world model that takes in hidden state + chosen action and outputs predicted goal location and the next state
    prediction = Chain(Dense(Nhidden+Naction, Npred_out, relu), Dense(Npred_out, Npred_out))

    return ModularModel(rnn, π_V, prediction)
end

# Flux.@functor ModularModel

