using PyCall
pyimport("d4rl")

function h5toEB(filename)
    # Load the data from the file
    actions = h5read(filename, "actions")
    obs = h5read(filename, "observations")
    rewards = h5read(filename, "rewards")
    dones = Bool.(h5read(filename, "terminals"))
    timeouts = Bool.(h5read(filename, "timeouts"))
    
    # we are going to remove all of the observations that included a terminal or done signal
    keep = .!(dones .| timeouts)
    keep[end] = false # remove the last one regardless
    
    
    next_obs = obs[:, 2:end][:, keep[1:end-1]] # shift the observations and then keep the ones that we want
    # Keep the rest of the elements consistent
    obs = obs[:,keep]
    actions = actions[:,keep]
    rewards = reshape(rewards[keep], 1, :)
    dones = reshape(dones[keep], 1, :)

    ExperienceBuffer(Dict(:s => obs, :sp=>next_obs, :a=>actions, :r=>rewards, :done=>dones))
end


