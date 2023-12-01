# using MetaRL
using Flux, Statistics, Random, Distributions, Zygote, StatsFuns, ChainRulesCore

function compute_useful_variables(Larena, Naction, Lplan)
    Nstates = Larena^2
    Nwall_in = 2 * Nstates # number of dimensions encoding wall location for agent input
    Nplan_in = 4 * Lplan + 1 # 4(actions) * planning horizon + 1(if goal loc found during planning)
    Nin = Naction + 1 + 1 + Nstates + Nwall_in + Nplan_in # 5 actions, 1 reward, 1 time, L^2 states, walls, & additional planning inputs
    Nout = Naction + 1 + Nstates + Nstates # 5(policy), 1(value function) & predicted next state and goal loc
    Npred_out = Nout - Naction - 1 # model output for next state and goal loc prediction
    return Nstates, Nplan_in, Nin, Nout, Npred_out
end

function evaluate_model(model, model_properties, # model related parameters
                        Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
                        βₚ, βₑ, βᵥ, batch)
"""
Evaluate the model during training.
t_rewd: total rewards for a batch of epsidoes
f_pred_next_loc: fraction of correct predictions of agent's next location and goal location
f_planning: fraction of planning in an episode 
""" 
    b_rep = 5 # when evaluating the model, let the agent repeat many batches to have a more stable read of its performance 
    t_rewd = zeros(b_rep, batch)
    f_pred_next_loc = zeros(b_rep, batch)
    f_planning = zeros(b_rep, batch)
    agent_outputs_t, actions_idx_t, rewards_t = [], [], []
    # times_t, agent_locs_idx_t, plans_states_t, wall_loc_hot_t, goal_loc_idx_t, x_plans_t=  [], [], [], [], [], [] # to delete 

    for i in 1:b_rep 
        # run a batch of episodes
        L, agent_outputs, actions_idx, rewards, world_states, D = run_episodes(model, model_properties, # model related parameters
                                                                                                      Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
                                                                                                      βₚ, βₑ, βᵥ;
                                                                                                      batch = batch, calc_loss = false)
        agent_locs_idx = mapreduce((x) -> x.agent_loc_idx, (x, y) -> cat(x, y; dims=2), world_states)

        # calculate total rewards (batch, D) for each episode
        t_rewd[i,:] = sum(rewards .>=0.5; dims=2)  # total rewards for each batch; 

        # fraction of correct predictions of agent's next location and goal location in an episode
        c_pred_next_locs = zeros(Int32, batch, D-1) # store if the prediction is correct: correct = 1, wrong = 0, if agent is teleported or not active, set to be -1 
        for d in 1:(D-1)
            ŝʼ = agent_outputs[Naction+1+1 : Naction+1+Nstates, :, d] # (Nstates, batch, D)
            s̃ʼ = [actions_idx[b,d] != 0 ? argmax(ŝʼ[:,b]) : -1 for b in 1:size(ŝʼ,2)] # index of predicted next location, set to -1 if episode is not active at the time 
            sʼ = agent_locs_idx[:, d+1] # actual next location (batch, )
            r = rewards[:, d] # (batch, )
            for b in 1:batch
                if s̃ʼ[b] == sʼ[b] && r[b] == 0
                    c_pred_next_locs[b,d] = 1 # 1 if prediction is correct and not teleported
                elseif s̃ʼ[b] != sʼ[b] && s̃ʼ[b] != -1 && r[b] == 0 
                    c_pred_next_locs[b,d] = 0 # 0 is predictiong wrong + agent is active and not teleproted 
                elseif s̃ʼ[b] == -1 || r[b] == 1
                    c_pred_next_locs[b,d] = -1 # -1 if not active or teleported 
                else 
                    println("Something is wrong while evaluating the model")
                end
            end
        end
        f_pred_next_loc[i,:] = [count(c_pred_next_locs[b,:] .== 1) / (count(c_pred_next_locs[b,:] .== 1 ) + count(c_pred_next_locs[b,:] .== 0)) for b = 1:batch] # fraction of correct prediction for the next location

        # fraction of planning 
        f_planning[i,:] = sum(actions_idx .== 5, dims=2) ./ sum(actions_idx .> 0.5, dims=2) # (batch,)
        # to delete 
        push!(agent_outputs_t, agent_outputs)
        push!(actions_idx_t, actions_idx)
        push!(rewards_t, rewards)
        #=
        push!(times_t, times)
        push!(agent_locs_idx_t, agent_locs_idx)
        push!(plans_states_t, plans_states)
        push!(wall_loc_hot_t, wall_loc_hot)
        push!(goal_loc_idx_t, goal_loc_idx)
        push!(x_plans_t, x_plans)
        =#
    end
    return mean(t_rewd), mean(f_pred_next_loc), mean(f_planning), agent_outputs_t, actions_idx_t, rewards_t #, times_t, agent_locs_idx_t, plans_states_t, wall_loc_hot_t, goal_loc_idx_t, x_plans_t # to delete 
end

function train(; Larena=4, Naction=5,  # environment related parameters
    seed=1, Lplan=8, # planning horizon
    epi_duration=2e4, time_p_action=400, time_s_action=120, # time related parameters
    Nhidden=100, Lrate=1e-3, βₚ=0.5, βₑ=0.05, βᵥ=0.05, # model related parameters 
    batch_size=40, n_b_epoch=200, n_epochs=1001, # training related parameters
    save_every=50, load_model=false, load_fname="") # parameters related to save/load the model 
"""
Take a gradient step for every a batch of episodes ($batch_size). Take $n_b_epoch gradient steps in an epoch.  Save the model every 50 epochs. 
batch_size: size of a batch (number of episodes in a batch)
n_b_epoch: number of batches in an epoch 
Lrate: learning rate
"""
    Random.seed!(seed) # set reandom seed 
    t0 = time() # wallclock time for tracking training progress   

    # Arrays to keep track of performance across `epochs
    losses = [] # loss
    t_rewds = [] # total rewards in an episode averaged across batches 
    f_pred_next_locs = [] # fraction of correct prediction in an episode averaged across batches
    f_plannings = [] # fraction of planning among all decision points 
    epoch = 1

    # Compute some useful variables and define model properties
    Nstates, Nplan_in, Nin, Nout, Npred_out = compute_useful_variables(Larena, Naction, Lplan)
    greedy_actions, no_planning= false, false
    model_properties = ModelProperties(Nhidden, Lplan, Nin, Nplan_in, Nout, Npred_out, greedy_actions, no_planning)

    # Build the model (with initial parameter values) and initialize the optimiser for this model
    if load_model # If we load a previous model
        fname = "saved_models/" * load_fname

        # load the parameters and initialize the model 
        progress, hps, rnn, π_V, prediction = load_model(fname)
        model = ModularModel(rnn, π_V, prediction)
        optim = Flux.setup(Flux.Adam(Lrate), model)

        # load the learning curve/performance from the model
        losses, t_rewds, f_pred_next_locs, f_plannings = progress[1], progress[2], progress[3], progress[4]

        # start from the next epoch 
        epoch = length(t_rewds) + 1
    else
        model = build_model(Nin, Nhidden, Npred_out, Naction)
        # optim = Flux.setup(Flux.Adam(Lrate), model) # to delete 
        optim = ADAM(Lrate) 
        loss_4_gradient = () -> loss_function(model, model_properties, # model related parameters
        Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
        βₚ, βₑ, βᵥ, batch_size) # training related parameters
        prms =  Params(Flux.params(model.rnn, model.π_V, model.prediction)) 
        # gs_list = [] # test, to delete 
    end 
    mod_name = name_the_model(Nhidden, seed, Lplan) 

    # to delete 
    agent_outputs_t, actions_idx_t, rewards_t, times_t, agent_locs_idx_t, plans_states_t, wall_loc_hot_t, goal_loc_idx_t, x_plans_t = [], [], [], [], [], [], [], [], []

    while epoch <= n_epochs
        flush(stdout) #flush output
        # For each epoch, run many batches (take one gradient step after each batch) 
        loss_epoch = [] # loss of each batch in an epoch 
        each_batch = 1 # new add  
        for each_batch in 1:n_b_epoch
            loss, gs = withgradient(loss_4_gradient, prms)
            # push!(gs_list, gs) to delete 
            push!(loss_epoch, loss)
            Flux.Optimise.update!(optim, prms, gs)
        
            #= Calculate loss and gradient of the loss
            loss, grads = Flux.withgradient(model) do m
                loss_function(m, model_properties, # model related parameters
            Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
            βₚ, βₑ, βᵥ, batch_size) # training related parameters
            end 
            
            # Update parameters
            push!(loss_epoch, loss)
            push!(gs_list, grads[1]) # test, to delete 
            Flux.update!(optim, model, grads[1])
            =# 
            each_batch =+ 1 # to add 
        end 
            
        # Evaluate the model, print & plot progress
        t_rewd, f_pred_next_loc, f_planning, agent_outputs, actions_idx, rewards = evaluate_model(model, model_properties, # model related parameters
                                                             Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
                                                             βₚ, βₑ, βᵥ, batch_size) # to delete 
                                      
        Flux.reset!(model) 
        push!(losses, mean(loss_epoch)) # append average loss of an epoch 
        push!(t_rewds, t_rewd)
        push!(f_pred_next_locs, f_pred_next_loc)
        push!(f_plannings, f_planning)
        elapsed_time = round(time() - t0; digits=1)
        println("epoch=$epoch t=$elapsed_time")
        plot_progress(losses, t_rewds, f_pred_next_locs, f_plannings) 

        # Save the model occasionally
        if epoch % save_every == 0
            Base.Filesystem.mkpath("saved_models_debug") # debug sepcific 
            filename = "saved_models/" * mod_name * "_" * string(epoch)
            progress = [losses, t_rewds, f_pred_next_locs, f_plannings]
            save_model(model, model_properties, optim, filename, progress, βₚ, βₑ, βᵥ, Larena, time_p_action, time_s_action)
        end

        epoch += 1 
        # to delete 
        push!(agent_outputs_t, agent_outputs)
        push!(actions_idx_t, actions_idx)
        push!(rewards_t, rewards)
        #=
        push!(times_t, times)
        push!(agent_locs_idx_t, agent_locs_idx)
        push!(plans_states_t, plans_states)
        push!(wall_loc_hot_t, wall_loc_hot)
        push!(goal_loc_idx_t, goal_loc_idx)
        push!(x_plans_t, x_plans)
        =# 
    end

    # Save model 
    Base.Filesystem.mkpath("saved_models")
    filename = "saved_models/" * mod_name * "_" * string(epoch-1)
    progress = [losses, t_rewds, f_pred_next_locs, f_plannings]

    return save_model(model, model_properties, optim, filename, progress, βₚ, βₑ, βᵥ, Larena, time_p_action, time_s_action) #= agent_outputs_t, actions_idx_t, rewards_t =#
end 


