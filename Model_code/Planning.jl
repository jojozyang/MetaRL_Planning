function plan(model, world_state, agent_output, h_rnn, Lplan, plan_idx, Naction, Nstates, Nin, Nplan_in)
    """
    Run planning algorithm if sampled action = 5.
    plan_states: (Lplan, batch), locations (idx) visited by agent during planning
    path_hot: (4, Lplan, batch), actions taken during plannign
    all_Vs: (batch,), store latest value function
    found_goal: (batch,) if goal is found
    """
        # extract some variables
        ĝ = agent_output[(Naction + Nstates + 2):(Naction + Nstates + 1 + Nstates), :] # (Nstates, batch)
        batch = size(ĝ, 2)
        g̃ = Int32.([argmax(ĝ[:, b]) for b in 1:batch]) # (batch, ) goal_location index representation in internal world model (stay the same during planning)
        wall_loc_hot = world_state.wall_loc_hot
        goal_loc_idx = world_state.goal_loc_idx
        time = world_state.time
    
        # initiate output arrays
        path_hot = zeros(Int32, 4, Lplan, batch) # store actions taken, always has batch throughout planning
        all_Vs = zeros(Float32, batch) # value functions, always has batch throughout planning
        plan_states = zeros(Int32, Lplan, batch) # location (idx) visited by agent during planning, always has batch throughout planning
        found_goal = zeros(Int32, batch) # if goal is found, always has batch throughout planning
    
        # only consider planning episodes
        wall_loc_hot = wall_loc_hot[:, :, plan_idx]
        goal_loc_idx = goal_loc_idx[plan_idx]
        time = time[plan_idx]
        h_rnn = h_rnn[:, plan_idx]
        g̃ = g̃[plan_idx] # (length_of_plan_idx, )
        ytemp = h_rnn # same for GRU
    
        x = zeros(Float32, Nin) # instantiate? Purpose?
    
        for p_steps = 1:Lplan
            p_batch = length(g̃) # number of ongoing episodes (goal is not reached during planning)
            # forward pass through recurrent part of the NN (starting from step 2)
            if p_steps > 1.5 # for the 1st planning step, use the current hidden state and input of the model
                h_rnn, ytemp = model.rnn[GRUind].cell(h_rnn, x)
            end
    
            # generate actions from hidden activity
            p_v = model.π_V(ytemp)
            π = p_v[1:4, :] # only consider physical actions
            logπ = logsoftmax(π; dims = 1) # compute the log of softmax
            Vs = p_v[Naction + 1, :]
            action_idx = sample_action(π)
            action_hot = idx_to_hot_action(action_idx, Naction)
    
            # store actions
            for (ib, b) = enumerate(plan_idx)
                path_hot[action_idx[1, ib], p_steps, b] = 1f0 # 'action_idx' contains ongoing planning episodes, 'path' contains all active episodes
            end
    
            # generate predictions
            ŝʼ = model.prediction([ytemp; action_hot])[1:Nstates, :] # predicted next state (Nstate, p_batch)
    
            # internal model representation of agent's next location
            ŝʼ = logsoftmax(ŝʼ; dims=1) # softmax over states
            s̃ʼ = Int.(argmax.([ŝʼ[:, b] for b = 1:p_batch])) # agent's internal world model representation of the next step
    
            # check if goal location is found during planning
            finished = findall(g̃ .== s̃ʼ)
            not_finished = findall(g̃ .!= s̃ʼ)
    
            # store some info
            all_Vs[plan_idx] = Vs # store latest value
            plan_states[p_steps, plan_idx] = s̃ʼ # store states
            found_goal[plan_idx[finished]] .= 1
    
            if length(not_finished) == 0
                return plan_states, path_hot, all_Vs, found_goal
            end
    
            # continue to plan if mental goal location is not reached
            wall_loc_hot = wall_loc_hot[:, :, not_finished]
            goal_loc_idx = goal_loc_idx[not_finished]
            s̃ʼ = s̃ʼ[not_finished]
            time = time[not_finished]
            plan_idx = plan_idx[not_finished]
            x_plan = zeros(Int32, Nplan_in, length(not_finished))
            action_hot = action_hot[:, not_finished]
            reward = zeros(Int32, 1, length(not_finished))
            h_rnn = h_rnn[:, not_finished]
            g̃ = g̃[not_finished]
    
            # update world_state
            world_state = WorldState(wall_loc_hot = Int32.(wall_loc_hot), goal_loc_idx = Int32.(goal_loc_idx),
                                     agent_loc_idx = Int32.(s̃ʼ), time = Int32.(time), x_plan = Int32.(x_plan), plan_states = Int32.(plan_states))
    
            # generate input
            x = generate_input(world_state, action_hot, reward, Nstates)
        end
        return plan_states, path_hot, all_Vs, found_goal
end