# Semi-Batch Reverse Osmosis Environment

A Julia-based simulation environment for Semi-Batch Reverse Osmosis (SBRO) systems, designed for reinforcement learning applications.

## Overview

This environment simulates a dynamic semi-batch reverse osmosis system with the following key features:

- Dynamic modeling of membrane fouling
- Temperature and concentration dependent fluid properties
- Support for both continuous circulation and purge modes
- Configurable system parameters and operating conditions
- HTTP JSON API for external control

## Project Structure

- `core/`: Core SBRO system models and simulation components
  - `SemiBatchReverseOsmosis.jl`: Main module
  - `SemiBatchReverseOsmosisSystem.jl`: SBRO system equations
  - `simulation_components.jl`: Helper functions for simulation

- `rl_env_components/`: Reinforcement learning environment components
  - `rl_env.jl`: Main RL environment interface
  - `rl_env_construction.jl`: Environment construction utilities
  - `rl_scenario.jl`: Scenario generation for feed conditions

- `rl_plans/`: Implementation of different control strategies
  - `plan_A.jl`: Basic control strategy implementation

- `testNbenchmarks/`: Test cases and benchmarking utilities

## Dependencies

Main dependencies include:
- ModelingToolkit
- OrdinaryDiffEq
- Unitful
- HTTP
- JSON3

## Usage

### Direct Julia Interface

```julia
# Initialize environment
dt = 30.0  # timestep in seconds
τ_max = 86400.0  # maximum episode length in seconds
sbro_env = initialize_sbro_env(; dt=dt, τ_max=τ_max)

# Set scenario conditions
reset_scenario!(sbro_env, 
    [15.0, 1.0, 0.05, 0.01],  # scenario parameters
    [60*60*8.0, 60*60*12.0, 12.0, 16.0]  # objective parameters
)

# Reset and run environment
exp_reset = reset!(sbro_env, nothing, nothing; dt=dt)
exp_step = step!(sbro_env, [5.0, 0.5, 0.0])  # [Q₀, R_sp, mode]
```
### HTTP Server Interface
```bash
julia -t N_THREADS http_json_api_loop.jl
```
TIP: increasing N_THREADS from 1 to 2 greatly improves the performance, even though core ODE solving is not affected by N_THREADS. Maybe it is due to async IO handling of HTTP.jl.

The server provides endpoints for:

- POST /reset_scenario: Set scenario conditions
- POST /reset: Reset environment
- POST /step: Execute environment step
- GET /render: Get environment state
