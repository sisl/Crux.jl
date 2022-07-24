using Crux, POMDPGym, POMDPs, POMDPPolicies, Flux, Random, BSON
using LocalApproximationValueIteration, GridInterpolations, LocalFunctionApproximation
Crux.set_function("isfailure", POMDPGym.isfailure)

# Creat the mdp
mdp = CollisionAvoidanceMDP()
S = state_space(mdp, μ=[0f0, 0f0, 0f0, 20f0], σ = [200f0, 10f0, 5f0, 20f0])

# Function to evaluate the policy
function evaluate_policy(pol; Neps = Int(1e5), S = state_space(mdp))
    s = Sampler(mdp, pol, A = DiscreteSpace(actions(mdp)), required_columns=[:fail, :x], S=S)
    D = episodes!(s, Neps=Neps)
    fail_rate = sum(D[:fail]) / Neps
    alert_rate = sum(D[:r] .== -1.0) / Neps
    println("fail_rate: $fail_rate, alert_rate: $alert_rate")
    return fail_rate, alert_rate
end


results = []

for i=1:5
    Random.seed!(i)


    # DQN_params
    N = 100000
    π_explore(outputs) = ϵGreedyPolicy(Crux.LinearDecaySchedule(1.0, 0.1, floor(Int, N/2)), outputs)
    DQN_params() = (S=S, 
                    N=N, 
                    buffer_size=Int(1e5), 
                    buffer_init=1000, 
                    π_explore=π_explore(actions(mdp)), 
                    c_opt=(;batch_size=256, optimizer=Adam(1e-4)))
    ARL_params() = (DQN_params()..., 
                    x_c_opt=(;batch_size=256, optimizer=Adam(1e-4)), 
                    desired_AP_ratio=3)
    neg_QS(outputs) = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(outputs), x->-softplus(-x))), outputs)
    QS(outputs) = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu),Dense(64, length(outputs))), outputs)
    (;batch_size=256, optimizer=Adam(1e-4))

    # Train with DQN
    𝒮_dqn = DQN(π=neg_QS(actions(mdp)); DQN_params()...)
    π_dqn = solve(𝒮_dqn, mdp)

    # Train with ISARL
    # Pf(outputs) = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(outputs), (x) -> softplus(x))), outputs, (x) -> (mdp.px.p .* (x .+ 1f-3)) ./ sum(mdp.px.p .* (x .+ 1f-3), dims=1), true)
    Pf(outputs) = DiscreteNetwork(Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, length(outputs), (x) -> -softplus(-(x-2)))), outputs, (x) -> (mdp.px.p .* softmax(x)) ./ sum(mdp.px.p .* softmax(x), dims=1), true)
    xlogprobs = Float32.(log.(mdp.px.p))
    𝒮_isarl = Crux.ISARL_Discrete(;π=AdversarialPolicy(neg_QS(actions(mdp)), Pf(disturbances(mdp))), ARL_params()..., px=mdp.px, xlogprobs=xlogprobs)
    π_isarl = solve(𝒮_isarl, mdp)

    # Train with RARL
    𝒮_rarl = RARL_Discrete(;π=AdversarialPolicy(neg_QS(actions(mdp)), QS(disturbances(mdp)), π_explore(disturbances(mdp))), ARL_params()...)
    π_rarl = solve(𝒮_rarl, mdp)

    # Evaluate all of the policies
    Neval = Int(1e5)
    dqn_fr, dqn_ar = evaluate_policy(π_dqn, Neps = Neval, S=S)
    isarl_fr, isarl_ar = evaluate_policy(protagonist(𝒮_isarl.π), Neps = Neval, S=S)
    rarl_fr, rarl_ar = evaluate_policy(protagonist(𝒮_rarl.π), Neps = Neval, S=S)

    push!(results, (;dqn_fr, dqn_ar, isarl_fr, isarl_ar, rarl_fr, rarl_ar))
end
BSON.@save "collision_avoidance_results.bson" results

results

# No policy at all
no_alert_pol = FunctionPolicy((s)->0f0)

# Optimal policy with dynamic programming
opt_pol = OptimalCollisionAvoidancePolicy(mdp)

evaluate_policy(opt_pol, Neps = Int(1e4))
evaluate_policy(no_alert_pol, Neps = Int(1e4))

