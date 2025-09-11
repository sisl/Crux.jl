# Installation

To install the package, run:
```
] add Crux
```

To edit or contribute use `] dev Crux` and the repo will be cloned to `~/.julia/dev/Crux`

## Usage with POMDPGym

The [POMDPGym](https://github.com/ancorso/POMDPGym) package provides a wrapper for [Gymnasium](https://gymnasium.farama.org/) environments for reinforcement learning to work with POMDPs.jl. Includes options to get the observation space from pixels.

* Install `POMDPGym` via:
```
] add https://github.com/ancorso/POMDPGym.jl
```
* The Python dependencies [gymnasium](https://gymnasium.farama.org/) and `pygame` will be automatically installed during the build step of this package.

### Atari and other environments
Currently, the automatic installation using `Conda.jl` does not install the Atari environments of Gymnasium. To do this, install Atari environments in a custom Python environment manually and ask `PyCall.jl` to use it. To elaborate, create a new Python virtual environment and run
```
pip install gymnasium[classic-control] gymnasium[atari] pygame
pip install autorom
```
Then run the shell command `AutoROM` and accept the Atari ROM license. Now you can configure `PyCall.jl` to use your Python environment following the [instructions here](https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#specifying-the-python-version).

Optionally, you can also install [MuJoCo](http://www.mujoco.org/).
