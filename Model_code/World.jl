struct WorldState
    wall_loc_hot # (Nstates, 4, i_batch)
    goal_loc_idx # (Nstates, i_batch)
    agent_loc_idx # (Nstates, i_batch)
    time # (i_batch)
    x_plan # (Nplan_in, i_batch) additional input if the agent planned during previous decision point
    plan_states # (LPlan, i_batch)
end

function WorldState(; wall_loc_hot, goal_loc_idx, agent_loc_idx, time, x_plan, plan_states)
    return WorldState(wall_loc_hot, goal_loc_idx, agent_loc_idx, time, x_plan, plan_states)
end

function initialize_world(batch, Larena, Nplan_in, Naction, Lplan)
"""
Initialize the world states and other inputs (previous action and reward)
world_state: a object/instance of WorldState
action_hot: (Naction, batch)
reward: (1, batch)
"""
    ignore_derivatives() do 
        # Initialize the wall location
        wall_loc_hot = generate_mazes(batch, Larena)
        Nstates = Larena ^ 2

        # Initialize goal and agent location
        goal_loc_idx = zeros(batch)
        agent_loc_idx = zeros(batch)
        for b in 1:batch
            goal = rand(1:Nstates) # choose a random goal position
            goal_loc_idx[b] = goal
            s = rand(setdiff(Array(1:Nstates), [goal])) # make sure the agent doesn't start at goal location
            agent_loc_idx[b] = s
        end

        # Initialize planning related states
        x_plan = zeros(Nplan_in, batch)
        plan_states = zeros(Lplan, batch)

        # Instantiate a WorldState object
        world_state = WorldState(wall_loc_hot = Int32.(wall_loc_hot), goal_loc_idx = Int32.(goal_loc_idx),
                                agent_loc_idx = Int32.(agent_loc_idx), time = zeros(Int32, batch), # starting time is 0 (batch, )
                                x_plan = Int32.(x_plan), plan_states = plan_states) # no planning input

        # Generate initial action and reward values as agent's input
        action_hot = zeros(Int32, Naction, batch)
        reward = zeros(Int32, 1, batch)

        # Generate input for the model 
        x = generate_input(world_state, action_hot, reward, Nstates)  # (Nin, batch)

    return world_state, x
    end
end

function generate_input(world_state, action_hot, reward, Nstates)
    """
    Generate the input x (Nin, i_batch) for the agent
    x: (Nin, i_batch)
    """
    time = Int32.(transpose(world_state.time)) # (1, i_batch)
    agent_loc_hot = Int32.(idx_to_hot_loc(world_state.agent_loc_idx, Nstates))
    wall_loc_hot = Int32.(world_state.wall_loc_hot)
    wall_input = Int32.(get_wall_input(wall_loc_hot))
    x_plan = Int32.(world_state.x_plan)

    x = [action_hot; reward; time; agent_loc_hot; wall_input; x_plan]

    return x

end

function step!(model_properties, Larena, Naction, Nstates, world_state, agent_output, action_idx, model, h_rnn, active_idx, batch, epi_duration, time_p_action, time_s_action)
"""
Execute the decision (physical or mental actions) and update various values.
"""
    ignore_derivatives() do
        # Extract model properties 
        Nplan_in = model_properties.Nplan_in
        Lplan = model_properties.Lplan
        Nin = model_properties.Nin
        Nplan_in = model_properties.Nplan_in

        agent_loc_idx = world_state.agent_loc_idx
        agent_loc_coord = idx_to_coord_loc(agent_loc_idx, Larena)
        wall_loc_hot = world_state.wall_loc_hot
        goal_loc_idx = world_state.goal_loc_idx
        reward = zeros(Int32, 1, batch)

        # For physical actions, find the viable action that doesn't result in hitting the wall
        v_action_hot = zeros(Int32, Naction, batch) # to store viable actions
        for b in 1:batch
            a = action_idx[1, b]
            if a < 4.5 && !Bool(wall_loc_hot[agent_loc_idx[b], a, b]) # physical actions && actions that don't result in hitting the wall
                v_action_hot[a, b] = 1
            end
        end

        # For all actions, find agent's next location (location will not change for nonviable physical actions and if the agent decides to plan)
        sʼ_coord = agent_loc_coord + [v_action_hot[1:1, :] - v_action_hot[2:2, :]; v_action_hot[3:3, :] - v_action_hot[4:4, :]] # state transition function part 1
        sʼ_coord = Int.((sʼ_coord .+ Larena .- 1) .% Larena .+ 1) # state transition function part 2
        sʼ_idx = coord_to_idx_loc(sʼ_coord, Larena) # action of 1,2,3,4,5 correspond to move down, up, right, left

        # If agent moves to the goal location, update reward and teleport agent
        for b in 1:batch
            if sʼ_idx[b] == goal_loc_idx[b]
                reward[1, b] = 1
                goal = goal_loc_idx[b]
                sʼ_idx[b] = rand(setdiff(Array(1:Nstates), [goal])) # make sure the agent doesn't start at goal location
            end
        end

        # Update time for episodes that did not decide to plan
        new_time = world_state.time
        not_plan_idx = findall([action_idx[1,b] != 5 for b in 1:batch])
        new_time[not_plan_idx] .+= time_p_action # for episodes that took a physical action (including the actiosn taht resutl in hitting the wall)

        # Run planning algorithm if agent decides to plan in at least one episode and update  time
        plan_idx = findall([action_idx[1,b] == 5 for b in 1:batch]) # plan episodes, index into active episodes
        plan_states = zeros(Int32, Lplan, batch)
        x_plan = zeros(Int32, Nplan_in, batch)
        if length(plan_idx) > 0.5 # update planning input
            plan_states, path_hot, all_Vs, found_goal = plan(model, world_state, agent_output, h_rnn, Lplan, plan_idx, Naction, Nstates, Nin, Nplan_in)
            for b = 1:batch
                x_plan[:, b] = [path_hot[:, :, b][:]; found_goal[b]] # flattened array of the imagined action sequencew; found goal in planning?
            end     
            new_time[plan_idx] .+= time_s_action
        end

        # Check if the episode is still active (before time out)
        active_idx = findall(new_time .<= epi_duration) # index of active episodes. The case when time = epi_duratrion is included to include the last step into calculation  
        a_batch = length(active_idx) # number of active episodes in a batch

        # Update world_state (agent location and time)
        world_state = WorldState(wall_loc_hot = Int32.(wall_loc_hot), goal_loc_idx = Int32.(goal_loc_idx),
                                agent_loc_idx = Int32.(agent_loc_idx), time = Int32.(new_time), x_plan = Int32.(x_plan), plan_states = Int32.(plan_states))

        # Generate one-hot representation for actions 
        action_hot = idx_to_hot_action(action_idx, Naction)

        x = generate_input(world_state, action_hot, reward, Nstates)  # (Nin, batch)

        return world_state, Int.(reward), active_idx, a_batch  
    end
end