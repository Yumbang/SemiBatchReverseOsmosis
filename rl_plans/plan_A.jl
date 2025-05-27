using StatsBase

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
                                              env.sys.SEC]]

    P_m_in  = [extracted_observation[1] for extracted_observation in extracted_observations]
    P_m_out = [extracted_observation[2] for extracted_observation in extracted_observations]
    C_perm  = [extracted_observation[3] for extracted_observation in extracted_observations]
    SEC     = [extracted_observation[4] for extracted_observation in extracted_observations]

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
    τ_passed = env.dt
    
    e_total  = mean(SEC[1:end-1], weights(diff(solutions.t))) * Q_perm / 3600 * env.dt 
    # SEC_mean = mean(SEC[1:end-1], weights(diff(solutions.t)))

    V_perm_mixed = Q_perm / 3600 * env.dt
    C_perm_mixed = mean(C_perm[1:end-1], weights(diff(solutions.t)))
    permeate     = [V_perm_mixed, C_perm_mixed]

    return (base_observation, τ_passed, permeate, e_total)
end

function calculate_sparse_reward(
    env::SemiBatchReverseOsmosisEnv,
    is_truncated::Bool
)
    base_reward = 0.0
    
    penalty_truncation = 1.0
    penalty_τ          = 0.01   # (s)⁻¹
    penalty_SEC        = (0.005) / 3600.0 / 1000.0    # (kWh/m³)⁻¹ → (Ws/m³)⁻¹
    penalty_conc       = 5.0    # (kg/m³)⁻¹

    # Give an extra penalty if the episode is truncated
    if is_truncated
        base_reward -= penalty_truncation
    end

    # Give penalty according to exceeded time and SEC
    τ_exceed     = max(0.0, env.problem.tspan[2] - env.τ_obj)
    SEC_total    = env.E_total_cur / env.V_perm_cur
    conc_exceed  = max(0.0, env.C_perm_cur - 0.01)
    
    base_reward -= penalty_τ    * τ_exceed +
                   penalty_SEC  * SEC_total +
                   penalty_conc * conc_exceed

    @info "Penalty terms" τ_exceed SEC_total conc_exceed

    return base_reward
end

function calculate_dense_reward(
    env::SemiBatchReverseOsmosisEnv
)
    base_reward = 0.0

    # Give incentive according to permeate volume accomplished during a cycle
    # No matter how much is `calculate_dense_reward` called, total incentive is designed to be same (V_perm_obj)
    # However, because of γ, RL agent may be prompted to accomplish the reward later in the episode.
    incentive_V_perm = 0.1  # (m³)⁻¹

    base_reward     += incentive_V_perm * env.V_perm_cycle

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
                                     dt = dt, reset_tspan=true)
    
                                     
    # Solve SBRO ODE problem. DP5 is turned out to be most efficient, compared to Tsit5
    sol = solve(env.problem, DP5())

    env.last_u = deepcopy(sol.u[end])

    # Process solution into proper observation & reward    
    base_observation, _, permeate, e_total = process_solution(env, sol)

    # StatsBase's `mean` is ×6 faster than manual weighted mean operation
    env.C_perm_cur      = mean([env.C_perm_cur, permeate[2]], weights([env.V_perm_cur, permeate[1]]))
    env.V_perm_cur     += permeate[1]
    env.V_perm_cycle   += permeate[1]
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
                                     dt = dt, reset_tspan=false)
    
                                     
    # Solve SBRO ODE problem. DP5 is turned out to be most efficient, compared to Tsit5
    sol = solve(env.problem, DP5())

    # Process solution into proper observation & reward    
    base_observation, _, permeate, e_total = process_solution(env, sol)

    env.last_u = deepcopy(sol.u[end])

    # StatsBase's `mean` is ×6 faster than manual weighted mean operation
    env.C_perm_cur      = mean([env.C_perm_cur, permeate[2]], weights([env.V_perm_cur, permeate[1]]))
    env.V_perm_cur     += permeate[1]
    env.V_perm_cycle   += permeate[1]
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
        dense_reward  = calculate_dense_reward(env)
        reward       += sparse_reward + dense_reward
    else
        if is_CC_done
            dense_reward = calculate_dense_reward(env)
            reward += dense_reward
        end
    end

    env.mode_cur = mode_temp

    observation  = vcat(base_observation, [env.τ_obj - env.problem.tspan[2], env.V_perm_obj - env.V_perm_cur])
    env.state    = observation

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

# dt                  = 30.0u"s" |> ustrip
# # maximum_episode_len = 12.0u"minute" |> u"s" |> ustrip
# maximum_episode_len = 1.0u"d" |> u"s" |> ustrip

# sbro_env = initialize_sbro_env(;dt=(dt), τ_max=(maximum_episode_len))

# reset_scenario!(sbro_env, [15.0, 1.0, 0.05, 0.01], [60*60*8.0, 60*60*12.0, 12.0, 16.0])

# exp_array = []

# exp_reset = reset!(sbro_env, nothing, nothing; dt=dt)
# begin
#     for _ ∈ 1:60
#         push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
#         if exp_array[end][:terminated] | exp_array[end][:truncated]
#             break
#         end
#     end

#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end

#     for _ ∈ 1:60
#         push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
#         if exp_array[end][:terminated] | exp_array[end][:truncated]
#             break
#         end
#     end

#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end

#     for _ ∈ 1:60
#         push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
#         if exp_array[end][:terminated] | exp_array[end][:truncated]
#             break
#         end
#     end

#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     for _ ∈ 1:60
#         push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
#         if exp_array[end][:terminated] | exp_array[end][:truncated]
#             break
#         end
#     end

#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     for _ ∈ 1:60
#         push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
#         if exp_array[end][:terminated] | exp_array[end][:truncated]
#             break
#         end
#     end

#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
#     push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
#     if exp_array[end][:terminated] | exp_array[end][:truncated]
#         @info "Done!"
#     end
# end

# exp_array[1]
# sbro_env

# using Plots
# obs_plots = []

# for idx in 1:11
#     temp_plt = plot([elem[:observation][idx] for elem in exp_array], label="Observation $(idx)", framestyle=:box)
#     push!(obs_plots, temp_plt)
#     # display(temp_plt)
# end
# obs_plot = plot(obs_plots..., size=(1800, 1200), suptitle="Observation variables of Semi-Batch RO Environment")
# savefig(obs_plot, "obs_plot.pdf")


# sbro_env

# using BenchmarkTools, JSON, JSON3



# @benchmark JSON.json(exp_array[end])    # Time  (mean ± σ):   2.550 μs ± 25.529 μs
# @benchmark JSON3.write(exp_array[end])  # Time  (mean ± σ):   3.067 μs ± 238.027 ns

# @benchmark exp_reset = reset!($sbro_env, nothing, nothing; dt=dt)   # Time  (mean ± σ):   178.847 μs ± 179.361 μs
# @benchmark begin
#     exp_reset = reset!($sbro_env, nothing, nothing; dt=dt)
#     exp_step  = step!($sbro_env, [5.0, 0.5, 0.0])
# end # Time  (mean ± σ):   353.538 μs ± 238.476 μs


# sol = solve(sbro_env.problem, DP5())
# sbro_env.step_cur += 1
# sbro_env.action = [5.0, 0.5, 0.0]
# base_observation, τ_passed, permeate = process_solution(sbro_env, sol)

# @benchmark process_solution(sbro_env, sol)  # Time  (mean ± σ):   74.589 μs ±   3.863 μs

# """
# Getting parameters by vector is slower!
# """
# @benchmark sbro_env.problem.ps[sbro_env.sys.Q₀]     # Time  (mean ± σ):   440.855 ns ± 241.633 ns
# @benchmark sbro_env.problem.ps[sbro_env.sys.R_sp]   # Time  (mean ± σ):   419.441 ns ± 343.009 ns
# @benchmark sbro_env.problem.ps[[sbro_env.sys.Q₀, 
#                                 sbro_env.sys.R_sp]] # Time  (mean ± σ):   1.659 μs ± 134.789 ns


# """
# Getting variables by vector is faster!
# """
# @benchmark sol[sbro_env.sys.P_m_in]     # Time  (mean ± σ):   4.816 μs ± 10.315 μs
# @benchmark sol[sbro_env.sys.P_m_out]    # Time  (mean ± σ):   4.526 μs ± 10.693 μs
# @benchmark sol[sbro_env.sys.C_perm]     # Time  (mean ± σ):   4.607 μs ± 10.751 μs
# @benchmark sol[[sbro_env.sys.P_m_in,
#                 sbro_env.sys.P_m_out,
#                 sbro_env.sys.C_perm]]   # Time  (mean ± σ):   8.932 μs ± 22.861 μs



# C_perm = sol[sbro_env.sys.C_perm]
# dtmap  = diff(sol.t)
# @benchmark mean_C_perm = mean(C_perm[1:end-1], weights(dtmap))  # Time  (mean ± σ):   119.997 ns ± 145.953 ns