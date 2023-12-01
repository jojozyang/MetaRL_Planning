using Random

function neighbor(cell, dir, Larena)
    """ Generate nighboring cell's coordinates given a cell's coordinates """
        neigh = ((cell + 1 * dir .+ Larena .- 1) .% Larena) .+ 1
        return neigh
end
    
function walk(maze::Array, nxtcell::Vector, Larena, visited::Vector=[])
""" 
Walk to all 4 nighbors of a cell to remove walls if the neighboring cells have not been visited.
nxtcell: (2,)
"""
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)
        push!(visited, (nxtcell[1] - 1) * Larena + nxtcell[2]) # add to list of visited cells in index representation
        for idir in randperm(4) # for each neighbor in randomly shuffled list
            neigh = neighbor(nxtcell, dirs[idir], Larena) # compute where we end up in coordinate representation
            ind = (neigh[1] - 1) * Larena + neigh[2] # convert coordinate to index representation for the neighboring cell
            if ind ∉ visited #check that we haven't been there
                maze[nxtcell[1], nxtcell[2], idir] = 0.0f0 # remove a wall from a side of a cell
                maze[neigh[1], neigh[2], dir_map[idir]] = 0.0f0 # remove a wall from the opposite side of the neighboring cell
                maze, visited = walk(maze, neigh, Larena, visited)
            end
        end
        return maze, visited
end

function generate_a_maze(Larena)
"""
Generate a maze by removing walls from a (Larena*Larena = Nstates) arena initially with walls everywhere.
Final maze dimension is (Nstates, 4), encoding the one-hot reprensentation of the locations of the walls.
The 1st dimension is the index of the cell
The 2nd dimension represents four sizes of the cell
1 = wall, 0 = no wall
"""
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]] # up, down, right and left directions
    dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3) # map between opposition directions
    maze = ones(Float32, Larena, Larena, 4) # start with walls everywhere (walls at four sides of every cell)
                                         # the 3rd dimension represents if there's a wall at the four sizes of a cell/grid (1 = wall)
    cell = rand(1:Larena, 2) # (2, ) starting cell (each grid of the maze is a cell)'s coordinate
    maze, visited = walk(maze, cell, Larena) # walk and remove walls

    # remove a couple of additional walls to increase degeneracy (holes = number of walls to be removed)
    holes = Int(3 * (Larena - 3)) # holes=3 for Larena=4, holes=6 for Larena=5
    for _ in 1:holes
        # pick a random side, remove the wall of a random cell that has a wall at the chosen side
        a = rand(1:4) # pick 1 of 4 sides
        indices = findall(maze[:, :, a] .== 1) # a list of cells that have a wall at the chosen side (in coordinate representation)
        if length(indices) > 0.5 # check for the super unlikely event that there are no walls satisfying this
            cell = rand(indices)
            neigh = neighbor([cell[1]; cell[2]], dirs[a], Larena)
            maze[cell[1], cell[2], a] = 0.0f0 # remove a wall from a side of a cell
            maze[neigh[1], neigh[2], dir_map[a]] = 0.0f0 # remove a wall from the opposite side of the neighbor cell
        end
    end

    # reshape the maze
    maze = reshape(permutedims(maze, [2, 1, 3]), prod(size(maze)[1:2]), 4)

    return Float32.(maze)
end

function generate_mazes(batch, Larena)
"""
Generate all the maze matrices that encode wall locations for a batch of episodes
Wall_loc_hot (Nstates, 4, batch)
"""
    Nstates = Larena^2
    wall_loc_hot = zeros(Nstates, 4, batch)
    for b in 1:batch
        wall_loc_hot[:,:,b] = generate_a_maze(Larena)
    end
    return wall_loc_hot
end