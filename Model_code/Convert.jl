# Convert between different representations of the state/goal location, generate reward location (index) and wall locations for RNN input
using Flux, Statistics, Random, Distributions, StatsFuns

function hot_to_idx_loc(hot)
"""
Covert the one-hot represnetation to index representation for a location (goal or agent)
input loc: (Nstates, i_batch)
output loc: (i_batch,)
"""
    i_batch = size(hot)[2]
    return [argmax(hot[:, b]) for b in 1:i_batch]
end

function idx_to_hot_loc(idx, Nstates)
"""
Convert index representation to one-hot representation for a location (goal or agent)
input loc: (i_batch, )
output loc: (Nstates, i_batch)
"""
    i_batch = size(idx)[1]
    hot = zeros(Int32, Nstates, i_batch)
    for b in 1:i_batch
        hot[idx[b], b] = 1
    end
    return hot # don't take gradients of this
end

function coord_to_idx_loc(coord, Larena)
"""
Covert coordinate representation to index representation of 1 to Larena^2.
input loc: (2, i_batch)
output loc: (i_batch,)
"""
  return Larena * (coord[1, :] .- 1) + coord[2, :]
end

function idx_to_coord_loc(idx, Larena)
"""
Convert index representation to coordinate representation.
input loc: (i_batch,)
output loc: (2 x i_batch)
"""
    return [floor.((transpose(idx) .- 1) ./ Larena .+ 1); (transpose(idx) .- 1) .% Larena .+ 1]
end

function idx_to_hot_action(act, Naction)
"""
Convert index representation to one-hot representation for actions
input action : (1, i_batch)
output action: (Naction, i_batch)
"""
    ignore_derivatives() do 
        i_batch = size(act)[2]
        hot = zeros(Int32, Naction, i_batch)
        for b in 1:i_batch
            hot[act[1, b], b] = 1
        end
        return hot
    end
end

function hot_to_idx_action(act, Naction)
"""
Convert one-hot representation to index representation for actions
input action: (Naction, i_batch)
output action: (1, i_batch)
"""
    i_batch = size(act)[2]
    idx = [argmax(act[:, b]) for b in 1:i_batch]
    return transpose(idx)
end

function get_wall_input(wall_loc_hot)
"""
Format wall location for agent's input
wall_loc_hot: (Nstates, 4, i_batch)
wall_input: (Nstates*2, i_batch)
"""
    wall_input = [wall_loc_hot[:, 1, :]; wall_loc_hot[:, 3, :]] # all horizontal and all vertical walls (1 is up, 3 is right)
    return wall_input # don't take gradients of this
end