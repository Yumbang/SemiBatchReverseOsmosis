include("../rl_plans/plan_A.jl")

dt                  = 30.0u"s" |> ustrip
maximum_episode_len = 1.0u"d" |> u"s" |> ustrip

sbro_env = initialize_sbro_env(;dt=(dt), τ_max=(maximum_episode_len))

reset_scenario!(sbro_env, [15.0, 1.0, 0.05, 0.01], [60*60*8.0, 60*60*12.0, 12.0, 16.0])

exp_array = []

exp_reset = reset!(sbro_env, nothing, nothing; dt=dt)
begin
    for _ ∈ 1:60
        push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
        if exp_array[end][:terminated] | exp_array[end][:truncated]
            break
        end
    end

    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end

    for _ ∈ 1:60
        push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
        if exp_array[end][:terminated] | exp_array[end][:truncated]
            break
        end
    end

    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end

    for _ ∈ 1:60
        push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
        if exp_array[end][:terminated] | exp_array[end][:truncated]
            break
        end
    end

    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    for _ ∈ 1:60
        push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
        if exp_array[end][:terminated] | exp_array[end][:truncated]
            break
        end
    end

    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    for _ ∈ 1:60
        push!(exp_array, step!(sbro_env, [5.0, 0.5, 0.0]))
        if exp_array[end][:terminated] | exp_array[end][:truncated]
            break
        end
    end

    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
    push!(exp_array, step!(sbro_env, [5.0, 0.5, 1.0]))
    if exp_array[end][:terminated] | exp_array[end][:truncated]
        @info "Done!"
    end
end

@show exp_array[1]
@show sbro_env

using Plots
obs_plots = []
obs_vars = [
    "T_feed"
    "C_feed"
    "C_pipe_c_out"
    "P_m_in"
    "P_m_out"
    "Q_circ"
    "Q_disp"
    "Q_perm"
    "C_perm"
    "τ_remaining"
    "V_perm_remaining"
]

for idx in 1:11
    temp_plt = plot([elem[:observation][idx] for elem in exp_array], label=obs_vars[idx], framestyle=:box)
    push!(obs_plots, temp_plt)
    savefig(temp_plt, "../env_figures/$(obs_vars[idx]).pdf")
    # display(temp_plt)
end
obs_plot = plot(obs_plots..., size=(1800, 1200), suptitle="Observation variables of Semi-Batch RO Environment")
savefig(obs_plot, "../env_figures/obs_plot.pdf")


sbro_env

using BenchmarkTools, JSON, JSON3



@benchmark JSON.json(exp_array[end])    # Time  (mean ± σ):   2.550 μs ± 25.529 μs
@benchmark JSON3.write(exp_array[end])  # Time  (mean ± σ):   3.067 μs ± 238.027 ns

@benchmark exp_reset = reset!($sbro_env, nothing, nothing; dt=dt)   # Time  (mean ± σ):   178.847 μs ± 179.361 μs
@benchmark begin
    exp_reset = reset!($sbro_env, nothing, nothing; dt=dt)
    exp_step  = step!($sbro_env, [5.0, 0.5, 0.0])
end # Time  (mean ± σ):   353.538 μs ± 238.476 μs


sol = solve(sbro_env.problem, DP5())
sbro_env.step_cur += 1
sbro_env.action = [5.0, 0.5, 0.0]
base_observation, τ_passed, permeate = process_solution(sbro_env, sol)

@benchmark process_solution(sbro_env, sol)  # Time  (mean ± σ):   74.589 μs ±   3.863 μs

"""
Getting parameters by vector is slower!
"""
@benchmark sbro_env.problem.ps[sbro_env.sys.Q₀]     # Time  (mean ± σ):   440.855 ns ± 241.633 ns
@benchmark sbro_env.problem.ps[sbro_env.sys.R_sp]   # Time  (mean ± σ):   419.441 ns ± 343.009 ns
@benchmark sbro_env.problem.ps[[sbro_env.sys.Q₀, 
                                sbro_env.sys.R_sp]] # Time  (mean ± σ):   1.659 μs ± 134.789 ns


"""
Getting variables by vector is faster!
"""
@benchmark sol[sbro_env.sys.P_m_in]     # Time  (mean ± σ):   4.816 μs ± 10.315 μs
@benchmark sol[sbro_env.sys.P_m_out]    # Time  (mean ± σ):   4.526 μs ± 10.693 μs
@benchmark sol[sbro_env.sys.C_perm]     # Time  (mean ± σ):   4.607 μs ± 10.751 μs
@benchmark sol[[sbro_env.sys.P_m_in,
                sbro_env.sys.P_m_out,
                sbro_env.sys.C_perm]]   # Time  (mean ± σ):   8.932 μs ± 22.861 μs



C_perm = sol[sbro_env.sys.C_perm]
dtmap  = diff(sol.t)
@benchmark mean_C_perm = mean(C_perm[1:end-1], weights(dtmap))  # Time  (mean ± σ):   119.997 ns ± 145.953 ns