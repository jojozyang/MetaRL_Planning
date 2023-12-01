using ChainRulesCore, Flux 
const GRUind = 1

function sample_action(policy_logits)
"""
Sample actions from the policy.
action_idx: (Nstates, i_batch)
"""
    ignore_derivatives() do # don't differentiate through this sampling process
        i_batch = size(policy_logits)[2] # i_batch can be a_bation or p_batch
        action_idx = zeros(Int32, 1, i_batch)
        π_sm = exp.(Float64.(policy_logits)) # probability of actions (up/down/right/left/stay)
        π_sm ./= sum(π_sm; dims=1) # normalize over actions to get softmax

        # set to argmax(log pi) if exponentiating gives infinitiy
        if any(isnan.(π_sm)) || any(isinf.(π_sm))
            for b in 1:i_batch
                if any(isnan.(π_sm[:, b])) || any(isinf.(π_sm[:, b]))
                    π_sm[:, b] = zeros(size(π_sm[:, b]))
                    π_sm[argmax(policy_logits[:, b]), b] = 1
                end
            end
        end

        action_idx[1,:] = Int32.(rand.(Categorical.([π_sm[:, b] for b = 1:i_batch])))

        return action_idx
    end
end

function forward_pass(model, x, h_rnn, Nstates, Naction)
"""
Generate output of the
h_rnn: (Nhidden, batch)
x: (Nin, batch) model input
agent_output: (Nout, batch)
action_idx: (1, batch)
"""
    # module 1 (hidden units)
    h_rnn, ytemp = model.rnn[GRUind].cell(h_rnn, x) # forward pass through recurrent part of the NN model

    # module 2 (policy and value function)
    p_v = model.π_V(ytemp) # (Naction+1, batch)
    π = p_v[1:Naction, :] # (Naction, batch) policy 
    logπ = logsoftmax(π; dims = 1) # compute the log of softmax
    V = p_v[(Naction + 1):(Naction + 1), :] # (1, batch)

    # sample an action from policy
    action_idx = sample_action(π) # (1, batch)
    action_hot = idx_to_hot_action(action_idx, Naction) # (Naction, batch)

    # module 3 (predicted next state and goal locations)
    predicted_output = model.prediction([ytemp; action_hot]) # (Nstates * 2, batch) predicted next state and goal locations (ŝʼ & ĝ)

    agent_output = [logπ; V; predicted_output]

    return agent_output, h_rnn, action_idx
end

function uniform_distribution(Naction, batch)
    @ignore_derivatives return ones(Naction, batch) / Naction # uniform distribution
end

function step_entropy_reg_loss(Naction, agent_output, batch, active_idx)
"""
Calculate the KL entropy regularization loss for one step (summed across episodes)
Goal: encourage exploration by minimizing entropy cost = increase divergence(KL) from uniform distribution (to decrease -KL)
agent_output: (Nout, batch)
Lₑ_step: scaler
"""
    U = uniform_distribution(Naction, batch)
    logU = log.(U)
    logπ = agent_output[1:Naction, :] # (Naction, batch)
    Lₑ_step = 0.0f0
    for b in active_idx
        Lₑ_step += sum(exp.(logπ[:,b]) .* (logU[:,b] - logπ[:,b])) # -KL[logπ || logU]
    end 
    return Lₑ_step
end

function step_prediction_loss(Nstates, Naction, agent_output, world_state, active_idx)
"""
Calculate cross-entropy loss between output distribution and true next state for one step/decision point (summed across episodes)
agent_output: (Nout, batch)
Lₚ_step: scaler
"""
    goal_loc_idx = world_state.goal_loc_idx
    next_state_idx = world_state.agent_loc_idx
    Lₚ_step  = 0.0f0
    ŝʼ = agent_output[(Naction + 1 + 1):(Naction + 1 + Nstates), :] # probability distributin of predicted next state (Nstates, batch)
    ĝ = agent_output[(Naction + Nstates + 2):(Naction + Nstates + 1 + Nstates), :]
    ŝʼ_sm = logsoftmax(ŝʼ; dims = 1)
    ĝ_sm = logsoftmax(ĝ; dims = 1)
    for b in active_idx
        Lₚ_step  -= ŝʼ_sm[next_state_idx[b], b] # -log(p(s))
        Lₚ_step  -= ĝ_sm[goal_loc_idx[b], b]
    end
    return Lₚ_step  #return summed loss
end

function advantage_function(rewards, agent_outputs, D, batch, Naction)
    """
    rewards (1, batch, D)
    agent_outputs (Nout, batch , D)
    δs (batch, D): advantage function, might contange missing value
    """
        rewards = rewards[1, :, :] # (batch, D)
        Vs = agent_outputs[Naction + 1, :, :] # value functions (batch, D)
        Rs = zeros(Int32, batch) # reward-to-go
        δs = zeros(Float32, batch, D) # advantage function: Rs - Vs
    
        for t in reverse(1:D) # Loop backward to calculate reward-to-go
            Rs = rewards[:, t] + Rs
            δs[:, t] = Rs - Vs[:, t]
        end
        return Float32.(δs) 
 end
 ChainRulesCore.@non_differentiable advantage_function(::Any...)

function zeropad(active_idx, action_idx, reward, x, batch) 
""" Set the values of finished episodes to be 0. """
    ignore_derivatives() do
        finished_idx = setdiff(1:batch, active_idx) 
        if length(finished_idx) > 0.5 
            x[:, finished_idx] .= 0f0
            reward[:, finished_idx] .= 0f0
            action_idx[:, finished_idx] .= 0f0
        end
        return x, reward, action_idx
    end
end

function run_episodes(model, model_properties, # model related parameters
    Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
    βₚ, βₑ, βᵥ;
    batch = batch_size, calc_loss = true)
"""
Roll out the policy and run a batch of episodes. This can be run during training the neural network, evaluatuing the model or apply the trained model.
During training, calc_loss is set to be true to calculate the loss in order to do gradient descent.
At each decision point, for active episodes, the agent will decide to either take physical actions or plan
L: sclaer, loss summed across all episodes and decision points
"""
    # Extract model properties 
    Nhidden = model_properties.Nhidden
    Lplan = model_properties.Lplan
    Nplan_in = model_properties.Nplan_in
    Nout = model_properties.Nout

    # Initialize some variables and arrays
    Lₚ = 0.0f0
    Lₑ = 0.0f0
    L = Float32(0.0) # loss of actor, critic, internal model prediction + entropy regularization
    agent_outputs = Array{Float32}(undef, Nout, batch, 0)
    actions_idx = Array{Int32}(undef, 1, batch, 0) # for storing action index across steps
    rewards = Array{Int32}(undef, 1, batch, 0) # for storing reward across steps, 1=reward, 0=no reward 
    hs = Array{Int32}(undef, Nhidden, batch, 0)
    world_states = Array{Int32}(undef, 0)
    h_rnn = model.rnn[1].cell.state0 .+ Float32.(zeros(Nhidden, batch)) # project initial hidden state to batch_size
    active_idx = [i for i in 1:batch] # idx of episodes in a batch that are not finished
    a_batch = length(active_idx) # number of active episodes in a batch
    D = 0 # decision point

    # Initialize the world: generate initial action, reward and world_state(wall, goal & agent locations, time, x_plan, plan_states)
    world_state, x  = initialize_world(batch, Larena, Nplan_in, Naction, Lplan)
   
    while a_batch > 0.5 # run until all episodes in the batch are finished
        ignore_derivatives() do 
            hs = cat(hs, h_rnn; dims=3)
            world_states = [world_states; world_state]
        end 

        # Take a forward pass of the model and generate outputs(policy, value function, predicted goal location and next state) hidden state, and action
        agent_output, h_rnn, action_idx  = forward_pass(model, x, h_rnn, Nstates, Naction)
    
        # Execute the decision (physical or mental actions) and update various values
        world_state, reward, active_idx, a_batch  = step!(model_properties, Larena, Naction, Nstates, world_state, agent_output, action_idx, model, h_rnn, active_idx, batch, epi_duration, time_p_action, time_s_action)

        # continue to calculate and store values if there are active episodes
        if a_batch > 0.5
            D += 1
            if calc_loss
                Lₑ += step_entropy_reg_loss(Naction, agent_output, batch, active_idx) # Sum entropy regularization loss across steps
                Lₚ += step_prediction_loss(Nstates, Naction, agent_output, world_state, active_idx)  # Sum prediction loss across steps
            end 
            
            # values are assigned to 0 if the episodes are finshed (missing data type does not work well with Zygote when taking gradient so use zero-padding instead)
            x, reward, action_idx = zeropad(active_idx, action_idx, reward, x, batch) 

            # Store agent outputs, rewards and actions 
            agent_outputs = cat(agent_outputs, agent_output; dims=3) # store output (Nout, batch, D)
            rewards = cat(rewards, reward; dims=3) # (1, batch, D)
            actions_idx = cat(actions_idx, action_idx; dims=3) # (1, batch, D)
        end
    end
    
    if calc_loss 
        # Calculate loss for actor and critic
        δs = advantage_function(rewards, agent_outputs, D, batch, Naction) # (batch, D)
        for t = 1:D
            active = Float32.(actions_idx[1, :, t] .> 0.5) #zero for finished episodes
            critic = δs[:, t] .* agent_outputs[Naction + 1, :, t] .* active # δ * V; critic (batch,) 
            L -= sum(βᵥ * (critic))
            for b in findall(Bool.(active)) 
                    actor = δs[b, t] * agent_outputs[actions_idx[1, b, t], b, t] # δ * logπ(a)
                    L -= actor
            end
        end
    
        # Sum up all the loss
        L += βₚ * Lₚ # add predictive loss for internal world model
        L -= βₑ * Lₑ # minus negative KL divergence (increase KL divergence)
        L /= batch # normalize by batch
    end    
    return L, agent_outputs, actions_idx[1, :, :], rewards[1, :, :], world_states, D
end
    
function loss_function(model, model_properties, # model related parameters
    Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
    βₚ, βₑ, βᵥ, batch_size) # training related parameters
""" The loss function for Flux to calculate the gradient """

    loss = run_episodes(model, model_properties, # model related parameters
    Larena, Nstates, Naction, epi_duration, time_p_action, time_s_action, # environment related parameters
    βₚ, βₑ, βᵥ;
    batch = batch_size, calc_loss = true)[1] 

    return loss
end

