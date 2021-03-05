using Crux, Flux, POMDPs, POMDPGym, Random, Plots
using BSON, TensorBoardLogger, StaticArrays, POMDPModels, Zygote


## Build the tasks
Random.seed!(1) # Set the random seed
Ntasks = 3 # Number of tasks to solver
sz = (7,5) # Dimension of the gridworld
input_dim = prod(sz)*3 # three channels represent player position, lava, and goal
tasks = [LavaWorldMDP(size = sz, tprob = 0.99, goal = :random, randomize_lava = false, num_lava_tiles = 6) for _=1:Ntasks]
S = state_space(tasks[1]) # The state space of the tasks
as = [actions(tasks[1])...] # The actions
render(tasks[3]) # Plots the task

## Training hyperparameters
out = "output/"
try mkdir(out) catch end
N = 10000 # Number of training steps for each task
eval_eps = 10 # Number os episodes used for evaluation of the policy

# Define the network
Q() = DiscreteNetwork(Chain(x->reshape(x, input_dim, :), Dense(input_dim, 64, relu), Dense(64,64, relu), Dense(64, 4)), as)

## from scratch
function from_scratch(;i, kwargs...)
    DQN(π=Q(), S=S, N=N, log=(dir=string(out,"log/scratch/task$i"), fns=[log_undiscounted_return(10, name="undiscounted_return/T$i")]))
end 
scratch_solvers = continual_learning(tasks, from_scratch)
BSON.@save string(out, "scratch_solvers.bson") scratch_solvers

## warm start
function warm_start(;i, solvers=[], tasks=[]) 
    # Copy over the previous policy 
    pol = isempty(solvers) ? Q() : deepcopy(solvers[end].π)
    
    # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
    samplers = [Sampler(t, pol, S) for t in tasks]
    
    # Construct the solver
    DQN(π=pol, S=S, N=N, log=(dir=string(out,"log/warmstart/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]))
end

warmstart_solvers = continual_learning(tasks, warm_start)
BSON.@save string(out,"warmstart_solvers.bson") warmstart_solvers



## Experience replay
experience_per_task=1000 # Number of samples to store for each task
experence_frac=0.5 # Fraction of the data that will come from past experience
bc_batch_size=64 # Batch size of the behavioral cloning regularization
λ_bc=1f0 # Behaviroal cloning regularization coefficient


function experience_replay(;i, solvers=[], tasks=[])
    # Copy over the previous policy 
    pol = isempty(solvers) ? Q() : deepcopy(solvers[end].π)
    
    # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
    samplers = [Sampler(t, pol, S) for t in tasks]
    
    # Experience replay
    experience, fracs = isempty(solvers) ? ([], [1.0]) : begin 
        s_last = samplers[end-1] # Sampler for the previous task (swap out with different samplers here)
        new_buffer = ExperienceBuffer(steps!(s_last, Nsteps=experience_per_task)) # sample trajectories from that task
        new_buffer.data[:value] = value(pol, new_buffer[:s]) # compute their values (for behavioral cloning regularization)
        [solvers[end].extra_buffers..., new_buffer], [1-experence_frac, experence_frac./ones(length(solvers))...]
    end
    
    bcreg = i>1 ? BCRegularizer(experience, bc_batch_size, device(pol), λ=λ_bc) : (x)->0
    
    # Construct the solver
    DQN(π=pol, S=S, N=N, log=(dir=string(out,"log/er/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]), 
        extra_buffers=experience,
        buffer_fractions=fracs,
        c_opt=(regularizer=bcreg,))
end

er_solvers = continual_learning(tasks, experience_replay)
BSON.@save string(out,"er_solvers.bson") er_solvers


## Elastic Weight consolidation
λ_fisher = 1e12 # regularization coefficient for ewc
fisher_batch_size = 128 # Batch size for approximating the fisher information
function ewc(;i, solvers=[], tasks=[])
    # Copy over the previous policy 
    pol = isempty(solvers) ? Q() : deepcopy(solvers[end].π)
    
    # Construct samplers for previous tasks (for recording the new policy performance on previous tasks)
    samplers = [Sampler(t, pol, S) for t in tasks]
    
    # Setup the regularizer
    reg = (x) -> 0
    i==2 && (reg = DiagonalFisherRegularizer(Flux.params(pol), λ_fisher)) # construct a brand new on
    i > 2 && (reg = deepcopy(solvers[end].c_opt.regularizer))
    if i>1
        loss = (𝒟) -> -mean(exp.(value(pol, 𝒟[:s])))
        update_fisher!(reg, solvers[end].buffer, loss, Flux.params(pol), fisher_batch_size) # update it with new data
    end
    
    # Construct the solver
    DQN(π=pol, S=S, N=N, log=(dir=string(out,"log/ewc/task$i"), fns=[log_undiscounted_return(samplers, eval_eps)]), 
        c_opt=(regularizer=reg,))
end
ewc_solvers = continual_learning(tasks, ewc)
BSON.@save string(out,"ewc_solvers.bson") ewc_solvers


## Plot the results

# load the results (optional)
scratch_solvers = BSON.load(string(out,"scratch_solvers.bson"))[:scratch_solvers]
warmstart_solvers = BSON.load(string(out,"warmstart_solvers.bson"))[:warmstart_solvers]
er_solvers = BSON.load(string(out,"er_solvers.bson"))[:er_solvers]
ewc_solvers = BSON.load(string(out,"ewc_solvers.bson"))[:ewc_solvers]


## Cumulative_rewards
p_rew = plot_cumulative_rewards(scratch_solvers, label="scratch", legend=:topleft)
plot_cumulative_rewards(warmstart_solvers, p=p_rew, label="warm start")
plot_cumulative_rewards(er_solvers, p=p_rew, label="experience replay")
plot_cumulative_rewards(ewc_solvers, p=p_rew, label="ewc", show_lines=true)
savefig(string(out,"cumulative_reward.pdf"))

## Jumpstart Performance
p_jump = plot_jumpstart(scratch_solvers, label="scratch", legend=:right)
plot_jumpstart(warmstart_solvers, p=p_jump, label="warm start")
plot_jumpstart(er_solvers, p=p_jump, label="experience replay")
plot_jumpstart(ewc_solvers, p=p_jump, label="ewc")
savefig(string(out,"jumpstart.pdf"))

## Peak performance
p_perf = plot_peak_performance(scratch_solvers, label="scratch", legend=:bottomleft)
plot_peak_performance(warmstart_solvers, p=p_perf, label="warm start")
plot_peak_performance(er_solvers, p=p_perf, label="experience replay")
plot_peak_performance(ewc_solvers, p=p_perf, label="ewc")
savefig(string(out,"peak_performance.pdf"))

## Steps to threshold
p_thresh = plot_steps_to_threshold(scratch_solvers, .99, label="scratch")
plot_steps_to_threshold(warmstart_solvers, .99, p=p_thresh, label="warm start")
plot_steps_to_threshold(er_solvers, .99, p=p_thresh, label="experience replay")
plot_steps_to_threshold(ewc_solvers, .99, p=p_thresh, label="ewc")
savefig(string(out,"steps_to_threshold.pdf"))

## Catastrophic forgetting
p_forget = Crux.plot_forgetting(scratch_solvers, label="scratch", legend=:bottomleft, size=(900,600))
Crux.plot_forgetting(warmstart_solvers, p=p_forget, label="warm start")
Crux.plot_forgetting(er_solvers, p=p_forget, label="experience replay")
Crux.plot_forgetting(ewc_solvers, p=p_forget, label="ewc")
savefig(string(out,"catastrophic_forgetting.pdf"))

