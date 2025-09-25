---
title: 'Crux.jl: Deep Reinforcement Learning in Julia'
tags:
  - Julia
  - deep reinforcement learning
  - imitation learning
  - batch reinforcement learning
  - adversarial reinforcement learning
  - continual learning
authors:
  - name: Robert J. Moss
    orcid: 0000-0003-2403-454X
    affiliation: 1
  - name: Anthony Corso
    orcid: 0000-0002-4027-0473
    affiliation: 1
  - name: Mykel J. Kochenderfer
    orcid: 0000-0002-7238-9663
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 25 September 2025
bibliography: paper.bib
header-includes: |
    \usepackage{listings}
    \usepackage{cleveref}
---
\lstdefinelanguage{Julia}{
    keywords=[3]{solve, actions, state_space, dim, relu, length},
    keywords=[2]{Crux, POMDPGym, GymPOMDP, DiscreteNetwork, Chain, Dense, ContinuousNetwork, REINFORCE, A2C, PPO, ActorCritic},
    keywords=[1]{using},
    sensitive=true,
    morecomment=[l]{\#},
    morecomment=[n]{\#=}{=\#},
    morestring=[s]{"}{"},
    morestring=[m]{'}{'},
    alsoletter=!?,
    literate={,}{{\color[HTML]{0F6FA3},}}1
             {\{}{{\color[HTML]{0F6FA3}\{}}1
             {\}}{{\color[HTML]{0F6FA3}\}}}1
}

\lstset{
    language         = Julia,
    backgroundcolor  = \color[HTML]{F2F2F2},
    basicstyle       = \small\ttfamily\color[HTML]{19177C},
    numberstyle      = \ttfamily\scriptsize\color[HTML]{7F7F7F},
    keywordstyle     = [1]{\bfseries\color[HTML]{1BA1EA}},
    keywordstyle     = [2]{\color[HTML]{0F6FA3}},
    keywordstyle     = [3]{\color[HTML]{0000FF}},
    stringstyle      = \color[HTML]{F5615C},
    commentstyle     = \color[HTML]{AAAAAA},
    rulecolor        = \color[HTML]{000000},
    frame=lines,
    xleftmargin=10pt,
    framexleftmargin=10pt,
    framextopmargin=4pt,
    framexbottommargin=4pt,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    showstringspaces=false,
    showspaces=false,
    showtabs=false,
    columns=fullflexible,
    keepspaces=true,
    numbers=none,
}


# Summary
\href{https://github.com/sisl/Crux.jl}{Crux.jl} is a Julia library for deep reinforcement learning (RL) that provides concise, modular implementations of widely used algorithms.
The package offers CPU/GPU-accelerated training using Flux.jl [@flux] and is built upon shared abstractions (policies, value functions, buffers, objectives, and update rules).
These abstractions helps with both code reuse and understanding the core differences between algorithms (e.g., their surrogate losses, trust-region constraints, or advantage estimations).
Crux.jl includes policy-gradient and actor-critic methods such as REINFORCE [@reinforce], PPO [@ppo], and TRPO [@trpo], along with off-policy value-based and actor–critic variants such as DQN [@dqn], TD3 [@td3], and SAC [@sac], with additional support for imitation, offline, adversarial, and continual learning algorithms.
Shown in \cref{fig:example}, the library integrates with POMDPs.jl [@pomdps_jl] and the Python gymnasium environments [@gym] for reproducible benchmarking and fast experimentation.

\begin{figure}
\begin{lstlisting}[language=Julia]
using Crux, POMDPGym

problem = GymPOMDP(:CartPole)
as = actions(problem)
S = state_space(problem)

# Flux actor and critic networks
A() = DiscreteNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(dim(S)..., 64, relu), Dense(64, 1)))

# Setup solvers and solve to get their respective policies
solver_reinforce = REINFORCE(S=S, π=A())
policy_reinforce = solve(solver_reinforce, problem)

solver_a2c = A2C(S=S, π=ActorCritic(A(), V()))
policy_a2c = solve(solver_a2c, problem)

solver_ppo = PPO(S=S, π=ActorCritic(A(), V()))
policy_ppo = solve(solver_ppo, problem)
\end{lstlisting}
\caption{Crux.jl usage for the cart-pole problem, solved using various deep RL algorithms.}\label{fig:example}
\end{figure}


# Statement of Need
Reinforcement learning libraries, such as Stable Baselines3 [@sb3] and RLlib [@rllib], often blur the distinction between algorithmic ideas and framework code, hindering fair comparison, reuse, and extension to settings such as partial observability, safety constraints, or offline data.
Crux.jl is a compact, Julia-native framework built on multiple dispatch and Flux.jl that factors training into explicit, swappable components such as policies, critics, buffers, return/advantage estimators, objectives, and update rules.
In contrast to the inheritance-based code for Stable Baseline3, Crux.jl implements the `DQN` solver using a simple `dqn_target` function for the `OffPolicySolver` type, and a separate `td3_target` function for the same `OffPolicySolver` when implementing `TD3`.
This design enables rigorous, reproducible experimentation across RL settings, and integration with POMDPs.jl and gymnasium standardizes environment interaction and evaluation.
In short, Crux.jl provides a principled, composable deep RL framework for Julia that enables rapid ablations, fair baselines, and reproducible results without sacrificing performance or clarity.


# Research and Industrial Usage
The design goals of Crux.jl are reflected in its use across a range of scientific and applied domains.
In aerospace, this package has been applied to energy-optimized path planning for unmanned aircraft in varying winds [@banerjee2024energy].
In computational physics, researchers have combined reinforcement learning with metaheuristics for Feynman integral reduction, demonstrating the method's role in symbolic and high-performance computing processes [@zeng2025reinforcement].
More broadly, Crux.jl has been used to prototype algorithms for validation of safety-critical systems [@valbook], where component-wise modularity and reproducibility are particularly valuable.


# Acknowledgments
The authors would like to thank the Stanford Center for AI Safety for funding this work.


# References