using StatsBase
using Printf
include("../rl_env_components/rl_env.jl");

"""
Process environment and solution 
"""
function process_solution(
    env :: SemiBatchReverseOsmosisEnv,
    solutions
)
    # Extract feed information from scenarios
    T_feed = env.scenarios[1][env.step_cur + 1]
    C_feed = env.scenarios[2][env.step_cur + 1]

    # Extract flow rate information from action
    Q₀, R_sp, mode  = env.action
    Q_perm          = Q₀ * R_sp
    Q_circ          = (mode == 0) ? Q₀ * (1 - R_sp) : 0.0
    Q_disp          = (mode == 1) ? Q₀ * (1 - R_sp) : 0.0

    # Extract the last unknowns (only the last one is used as observation)
    _, C_pipe_c_out, _, _ = solutions[end]

    # Extract required variables from solution
    extracted_observations       = solutions[[env.sys.P_m_in,
                                              env.sys.P_m_out,
                                              env.sys.C_perm,
                                              env.sys.SEC,
                                              env.sys.Q_feed]]

    P_m_in  = [extracted_observation[1] for extracted_observation in extracted_observations]
    P_m_out = [extracted_observation[2] for extracted_observation in extracted_observations]
    C_perm  = [extracted_observation[3] for extracted_observation in extracted_observations]
    SEC     = [extracted_observation[4] for extracted_observation in extracted_observations]
    Q_feed  = [extracted_observation[5] for extracted_observation in extracted_observations]

    # Construct base obervation variable
    base_observation = [
        T_feed
        C_feed
        C_pipe_c_out
        P_m_in[end]
        P_m_out[end]
        Q_circ
        Q_disp
        Q_perm
        C_perm[end]
    ]

    # Prepare processed returns
    
    e_total  = mean(SEC[1:end-1], weights(diff(solutions.t))) * Q_perm / 3600 * env.dt 
    # SEC_mean = mean(SEC[1:end-1], weights(diff(solutions.t)))

    V_perm_mixed = Q_perm / 3600 * env.dt
    V_fresh_feed = mean(Q_feed) / 3600 * env.dt
    C_perm_mixed = mean(C_perm[1:end-1], weights(diff(solutions.t)))
    permeate     = [V_perm_mixed, C_perm_mixed]

    return (base_observation, V_fresh_feed, permeate, e_total)
end

function calculate_sparse_reward(
    env::SemiBatchReverseOsmosisEnv,
    is_truncated::Bool
)
    base_reward = 0.0
    
    if isnothing(env.reward_conf)
        penalty_truncation = 1.0
        penalty_SEC        = (0.005) / 3600.0 / 1000.0    # (kWh/m³)⁻¹ → (Ws/m³)⁻¹
        penalty_conc       = 5.0    # (kg/m³)⁻¹
        incentive_termination = 1e-4
    else
        penalty_truncation = env.reward_conf[:penalty_truncation]
        penalty_SEC        = env.reward_conf[:penalty_SEC]
        penalty_conc       = env.reward_conf[:penalty_conc]
        incentive_termination = env.reward_conf[:incentive_termination]
    end

    # Give an extra penalty if the episode is truncated
    if is_truncated
        base_reward -= penalty_truncation
    else
        # Give an extra incentive if the agent completed the mission (Produce given amount of water, in given time!)
        if (env.problem.tspan[2] - env.τ_obj) < 0.0
            # base_reward += incentive_termination * (env.τ_obj - env.problem.tspan[2])
            base_reward += incentive_termination
        end
    end

    # Give penalty according to exceeded time and SEC
    # τ_exceed     = max(0.0, env.problem.tspan[2] - env.τ_obj) Moved to calculate_dense_reward (0.1.4)
    SEC_total    = env.E_total_cur / env.V_perm_cur
    conc_exceed  = max(0.0, env.C_perm_cur - 0.025)
    
    # base_reward -= penalty_τ    * τ_exceed +
    base_reward -= penalty_SEC  * SEC_total +
                   penalty_conc * conc_exceed

    @info "Penalty terms" (penalty_SEC * SEC_total) (penalty_conc * conc_exceed)

    return base_reward
end

function calculate_cycle_reward(
    env::SemiBatchReverseOsmosisEnv
)
    base_reward = 0.0

    # Give incentive according to permeate volume accomplished during a cycle
    # No matter how much is `calculate_cycle_reward` called, total incentive is designed to be same (V_perm_obj)
    # However, because of γ, RL agent may be prompted to accomplish the reward later in the episode.
    # Penalty is given with to the volume of disposed brine, promoting high-recovery operation.
    if isnothing(env.reward_conf)
        incentive_V_perm = 0.1  # (m³)⁻¹
        penalty_V_disp = 0.1    # (m³)⁻¹
        
    else
        incentive_V_perm = env.reward_conf[:incentive_V_perm]
        penalty_V_disp   = env.reward_conf[:penalty_V_disp]
        
    end

    base_reward += incentive_V_perm * env.V_perm_cycle - penalty_V_disp * env.V_disp

    return base_reward
end

function calculate_dense_reward(
    env::SemiBatchReverseOsmosisEnv, V_fresh_feed::Float64
)
    if isnothing(env.reward_conf)
        penalty_τ          = 1e-6   # (s)⁻¹
        penalty_V_perm     = 1e-3   # (-)
        penalty_V_feed     = 0.1    # (m³)⁻¹
    else
        penalty_τ          = env.reward_conf[:penalty_τ]
        penalty_V_perm     = env.reward_conf[:penalty_V_perm]
        penalty_V_feed   = env.reward_conf[:penalty_V_feed]
    end

    base_reward = 0.0

    if (env.problem.tspan[2] - env.τ_obj) > 0.0
        base_reward -= penalty_τ * env.dt
    end

    if (env.V_perm_cur > env.V_perm_obj)
        # base_reward -= (env.V_perm_cur - env.V_perm_obj)
        base_reward -= penalty_V_perm
    end

    base_reward -= penalty_V_feed * V_fresh_feed

    return base_reward
end

"""
Reset scenario and objective with random values given setting
`reset_scenario!` is designed to be called outside `reset!`, and prior to (i.e., in main loop).
"""
function reset_scenario!(
    env                 :: SemiBatchReverseOsmosisEnv, 
    scenario_condition  :: Vector{Float64},
    objective_condition :: Vector{Float64}
)
    dt          = env.dt
    len_episode = env.τ_max

    μ_T, σ_T, μ_C, σ_C = scenario_condition
    
    T_feed_scenario    = generate_T_feed_scenario(μ_T, σ_T; dt = dt, len_episode = len_episode)
    C_feed_scenario    = generate_C_feed_scenario(μ_C, σ_C; dt = dt, len_episode = len_episode)
    env.scenarios      = [T_feed_scenario, C_feed_scenario]
    env.step_max       = length(T_feed_scenario)

    low_τ_obj, high_τ_obj,
    low_V_perm_obj, high_V_perm_obj = objective_condition

    τ_obj, V_perm_obj = generate_objectives_scenario(low_τ_obj, high_τ_obj, low_V_perm_obj, high_V_perm_obj)
    env.τ_obj         = τ_obj
    env.V_perm_obj    = V_perm_obj

    return nothing
end

"""
Reset environment and return the first observation & reward
operation_condition is [Q₀, R_sp, mode], i.e., as same as action
"""
function reset!(
        env                 :: SemiBatchReverseOsmosisEnv, 
        operation_condition :: Union{Nothing, Vector{Float64}},
        u_initial           :: Union{Nothing, Vector{Float64}};
        dt::Float64
    )
    env.step_cur   = 0
    env.V_perm_cur = 0.0
    env.V_perm_cycle = 0.0
    env.V_disp = 0.0
    env.C_perm_cur = 0.0
    env.E_total_cur = 0.0
    env.E_total_cycle = 0.0
    env.mode_cur   = nothing
    env.cycle_cur  = 1
    env.last_u     = nothing
    env.dt         = dt # In case one wants to change dt
    env.episode   += 1

    # Parameter & Initial condition generation
    if isnothing(operation_condition)
        Q₀_new   = 5.0
        R_sp_new = 0.5
        mode     = 0.0
    else
        Q₀_new, R_sp_new, _ = operation_condition
        mode = 0.0
    end

    action     = [Q₀_new, R_sp_new, mode]
    env.action = action

    if isnothing(u_initial)
        _, default_initial_condition = default_parameters(env.sys; C_feed=env.scenarios[2][1], T_feed=env.scenarios[1][1], Q₀=Q₀_new, R_sp=R_sp_new)
        initial_condition_new        = default_initial_condition
    else
        initial_condition_new = u_initial
    end

    # SBRO ODE problem reconstruction
    op_var_new   = [Q₀_new, 0.0, env.scenarios[2][1], env.scenarios[1][1], R_sp_new]
    env.problem  = update_parameters(env.sys, 
                                     env.problem, 
                                     env.setp_handle, 
                                     op_var_new, 
                                     initial_condition_new; 
                                     dt = env.dt, reset_tspan=true)
    
                                     
    # Solve SBRO ODE problem. DP5 is turned out to be most efficient, compared to Tsit5
    sol = solve(env.problem, DP5())

    env.last_u = deepcopy(sol.u[end])

    # Process solution into proper observation & reward    
    base_observation, _, permeate, e_total = process_solution(env, sol)

    # StatsBase's `mean` is ×6 faster than manual weighted mean operation
    env.C_perm_cur      = mean([env.C_perm_cur, permeate[2]], weights([env.V_perm_cur, permeate[1]]))
    env.V_perm_cur     += permeate[1]
    env.V_perm_cycle   += permeate[1]
    env.V_disp         += base_observation[7] * env.dt / 3600.0
    env.E_total_cur    += e_total
    env.E_total_cycle  += e_total
    env.mode_cur        = :CC

    observation = vcat(base_observation, [env.τ_obj - env.problem.tspan[2], env.V_perm_obj - env.V_perm_cur])
    env.state = observation

    # return proper information
    experience = Dict([
        :observation => observation,
        :info => Dict([:episode => env.episode, :step => env.step_cur, :cycle => env.cycle_cur])
    ])

    return experience
end

"""
Step environment with the given action and return the transition.
Action is [Q₀, R_sp, mode].
"""
function step!(
    env::SemiBatchReverseOsmosisEnv,
    action::Vector{Float64}
)
    env.step_cur  += 1

    # Parameter & Initial condition generation
    Q₀_new, R_sp_new, mode = action

    env.action = action
    
    mode_temp       = (mode == 1.0) ? (:CC) : (:purge)

    is_purging_done        = (env.mode_cur == :purge) & (mode_temp == :CC)
    is_CC_done             = (env.mode_cur == :CC) & (mode_temp == :purge)

    if is_purging_done
        env.V_perm_cycle  = 0.0
        env.V_disp = 0.0
        env.E_total_cycle = 0.0
        env.cycle_cur += 1
    end

    # SBRO ODE problem reconstruction
    op_var_new   = [
                    Q₀_new,
                    mode, 
                    env.scenarios[2][env.step_cur+1], 
                    env.scenarios[1][env.step_cur+1], 
                    R_sp_new
                   ]

    env.problem  = update_parameters(env.sys, 
                                     env.problem, 
                                     env.setp_handle, 
                                     op_var_new, 
                                     env.last_u; 
                                     dt = env.dt, reset_tspan=false)
    
                                     
    # Solve SBRO ODE problem. DP5 is turned out to be most efficient, compared to Tsit5
    sol = solve(env.problem, DP5())

    # Process solution into proper observation & reward    
    base_observation, V_fresh_feed, permeate, e_total = process_solution(env, sol)

    env.last_u = deepcopy(sol.u[end])

    # StatsBase's `mean` is ×6 faster than manual weighted mean operation
    env.C_perm_cur      = mean([env.C_perm_cur, permeate[2]], weights([env.V_perm_cur, permeate[1]]))
    env.V_perm_cur     += permeate[1]
    env.V_perm_cycle   += permeate[1]
    env.V_disp         += base_observation[7] * env.dt / 3600.0
    env.E_total_cur    += e_total
    env.E_total_cycle  += e_total

    reward      = 0.0

    is_V_perm_accomplished = (env.V_perm_cur ≥ env.V_perm_obj)
    is_time_limit_hit      = (env.step_max == (env.step_cur + 1))
    # TODO: Add one more truncation condition about concentration

    is_terminated = (is_V_perm_accomplished & is_CC_done)
    is_truncated  = (is_time_limit_hit)

    if (is_terminated | is_truncated)
        env.episode  += 1
        sparse_reward = calculate_sparse_reward(env, is_truncated)
        cycle_reward  = calculate_cycle_reward(env)
        reward       += sparse_reward + cycle_reward
        @info "Episode $(env.episode) completed" (env.V_perm_cur/env.V_perm_obj) env.step_cur env.cycle_cur
    else
        if is_CC_done
            cycle_reward = calculate_cycle_reward(env)
            reward += cycle_reward
        end
    end

    dense_reward = calculate_dense_reward(env, V_fresh_feed)
    reward += dense_reward

    env.mode_cur = mode_temp

    observation  = vcat(base_observation, [env.τ_obj - env.problem.tspan[2], env.V_perm_obj - env.V_perm_cur])
    env.state    = observation

    check_observation_sanity(observation)

    env.reward   = reward    

    # return proper information
    experience = Dict([
        :observation => observation,
        :reward => env.reward,
        :terminated => is_terminated,
        :truncated => is_truncated,
        :info => Dict([:episode => env.episode, :step => env.step_cur, :cycle => env.cycle_cur])
    ])

    return experience
end

"""
Hard-reset the environment.
Designed to call at initialization phase of SBRO gymnasium environment.
"""
function hard_reset!(env::SemiBatchReverseOsmosisEnv; dt::Float64, τ_max::Float64)
    step_max = ceil(Int, τ_max/dt)

    env.reward_conf = nothing
    env.episode  = 0
    env.step_cur = 0
    env.step_max = step_max - 1
    env.τ_max    = τ_max
    env.τ_obj    = nothing
    env.V_perm_cur = nothing
    env.V_perm_cycle = nothing
    env.V_perm_obj = nothing
    env.V_disp = nothing
    env.C_perm_cur = nothing
    env.E_total_cur = nothing
    env.E_total_cycle = nothing
    env.mode_cur = nothing
    env.cycle_cur = nothing
    env.last_u = nothing
    env.scenarios = nothing
    env.action = nothing
    env.state = nothing
    env.reward = nothing

    return nothing
end

function render(env::SemiBatchReverseOsmosisEnv; mode=:text)
    if mode == :text
        return """
====== Environment Settings ======
dt    [s]: $(env.dt)
τ_max [s]: $(env.τ_max)

====== Objective Settings ======
τ_obj      [s] : $(env.τ_obj)
V_perm_obj [m³]: $(env.V_perm_obj)

====== Current Status ======
episode  : $(env.episode)
step_cur : $(env.step_cur) / $(env.step_max)
cycle_cur: $(env.cycle_cur)

τ_cur      [s]    : $(env.problem.tspan[2])
V_perm_cur [m³]   : $(env.V_perm_cur)
C_perm_cur [kg/m³]: $(env.C_perm_cur)
E_total_cur[Ws]   : $(env.E_total_cur)

V_perm_cycle  [m³]: $(env.V_perm_cycle)
E_total_cycle [Ws]: $(env.E_total_cycle)"""
    else
        return nothing
    end
end

function check_observation_sanity(observation)
    greater_than_zero = all(observation[1:9] .> -1e-5)
    @assert greater_than_zero "Negative output detected ($observation)." 
    # @assert false "Negative output detected."

    return nothing
end