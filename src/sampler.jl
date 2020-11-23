abstract type Sampler end

struct StepSampler
    mdp
    𝒮::Solver
    s = rand(𝒮.rng, initialstate(mdp))
    svec = convert_s(AbstractArray, s, mdp)
    episode_steps::Int64 = 0
end

struct EpisodeSampler
    mdp
    𝒮::Solver
    return_checker::Union{Function, Nothing} = nothing
    step_checker::Union{Function, Nothing} = nothing
end 


function step!(sampler::Sampler; explore = true, Nsteps = 1)
    data = mdp_data(sampler.𝒮.sdim, sampler.𝒮.adim, Nsteps)
    for i=1:Nsteps
        # Take a step
        a = explore ? action(sampler.𝒮.exploration_policy, sampler.𝒮.π, 𝒮.i, sampler.svec) : action(sampler.𝒮.π, sampler.svec)
        sp, r = gen(mdp, sampler.s, a, 𝒮.rng)
        done = isterminal(mdp, sp)
    
        # Save the tuple
        spvec = convert_s(AbstractArray, sp, mdp)
        data[:s][:,i] .= sampler.svec
        data[:a][:,i] .= (a isa AbstractArray) ?  a : Flux.onehot(a, actions(mdp))
        data[:sp][:,i] .= spvec
        data[:r][1, i] = r
        data[:done][1,i] = done
        
        # Cut the episode short if needed
        sampler.episode_steps += 1
        if done || sampler.episode_steps >= sampler.𝒮.max_steps 
            sampler.s = rand(𝒮.rng, initialstate(mdp))
            sampler.svec = convert_s(AbstractArray, sampler.s, mdp)
            sampler.episode_steps = 0
        else
            sampler.s = sp
            sampler.svec = spvec
        end
    end
    data
end

function push_episodes!(b::ExperienceBuffer, mdp, N; policy = RandomPolicy(mdp), rng::AbstractRNG = Random.GLOBAL_RNG, baseline = nothing, max_steps = 100)
    i = 0
    γ = Float32(discount(mdp))
    while i < N
        h = simulate(HistoryRecorder(max_steps = min(max_steps, N - i), rng = rng), mdp, policy)
        [push!(b, s, a, r, sp, isterminal(mdp, sp), mdp) for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")]
        Nsteps = length(h)
        if !isnothing(baseline)
            fill_gae!(b, b.next_ind, Nsteps, baseline.V, baseline.λ, γ)
            fill_returns!(b, b.next_ind, Nsteps, γ)
        end
        i += Nsteps
    end
end

# TODO: Create a new sampling file?
function gen_buffer(mdps, pol, N; desired_return = nothing, max_tries = 100*N, max_steps = 100, nonzero_transitions_only = false)
    b = ExperienceBuffer(mdps[1], N)
    i = 1
    while length(b) < N && i < max_tries
        mdp = mdps[mod1(i, length(mdps))]
        h = simulate(HistoryRecorder(max_steps = max_steps), mdp, pol)
        if isnothing(desired_return) || undiscounted_reward(h) ≈ desired_return
            for (s, a, r, sp) in eachstep(h, "(s, a, r, sp)")
                if !nonzero_transitions_only || r != 0
                    push!(b, s, a, r, sp, isterminal(mdp, sp), mdp)
                end
            end
        end
        i += 1
    end
    b
end

