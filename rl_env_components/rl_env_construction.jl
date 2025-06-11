mutable struct SemiBatchReverseOsmosisEnv
    const sys   :: ODESystem                # Semi-batch RO system (simplified)
    problem     :: ODEProblem               # Semi-batch RO ODE problem
    setp_handle :: SymbolicIndexingInterface.ParameterHookWrapper # Parameter updater
    reward_conf :: Union{Nothing, Dict{Symbol, Float64}}
    dt          :: Float64                  # dt for each environment step [s]. NOT THAT OF ODE!
    episode     :: Int                      # episode counter
    step_cur    :: Int                      # current step counter
    step_max    :: Union{Nothing, Int}      # maximum step
    τ_max       :: Union{Nothing, Float64}  # maximum time [s]
    τ_obj       :: Union{Nothing, Float64}  # objective time limit [s]
    V_perm_cur  :: Union{Nothing, Float64}  # current cumulative permeate volume [m³]
    V_perm_cycle:: Union{Nothing, Float64}  # cycle cumulative permeate volume [m³]
    V_perm_obj  :: Union{Nothing, Float64}  # objective permeate volume [m³]
    V_disp      :: Union{Nothing, Float64}  # disposed brine volume [m³]
    C_perm_cur  :: Union{Nothing, Float64}  # current cumulative permeate concentration [kg/m³]
    E_total_cur :: Union{Nothing, Float64}
    E_total_cycle :: Union{Nothing, Float64}
    mode_cur    :: Union{Nothing, Symbol}
    cycle_cur   :: Union{Nothing, Int}
    last_u      :: Union{Nothing, Vector{Float64}}
    scenarios   :: Union{Nothing, Vector{Vector{Float64}}} # T & C scenario
    action      :: Union{Nothing, Vector{Float64}}
    state       :: Union{Nothing, Vector{Float64}}
    reward      :: Union{Nothing, Float64}
end

function Base.show(io::Core.IO, env::SemiBatchReverseOsmosisEnv)
    print(io, """
    ====== System ======
    $(env.sys)

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
    E_total_cycle [Ws]: $(env.E_total_cycle)""")
end

"""
Initialize a new SemiBatchReverseOsmosisEnv.
None of the numbers (i.e., C_feed, T_feed, Q₀ etc.), except dt, are not designed to be used.
To configure actual sbro ODE problem, use `reset!` in desired [planX.jl] file.

dt: Δt for each environment step [s]
"""
function initialize_sbro_env(;dt::Float64, τ_max::Float64)
    dummy_C_feed = 0.05
    dummy_T_feed = 15.0

    simple_sbro  = make_simple_sbro_system();
    def_param    = default_parameters(simple_sbro; C_feed = dummy_C_feed, T_feed = dummy_T_feed);
    sbro_prob,
     setp_handle = make_sbro_problem(simple_sbro, def_param...; dt = 0.0);   # Dummy problem, tspan with no length
    step_max = ceil(Int, τ_max/dt)

    SBROEnv = SemiBatchReverseOsmosisEnv(
        simple_sbro,
        sbro_prob,
        setp_handle,
        nothing,
        dt,
        0,          # Every reset!() call increase episode by 1. I.e., episode must start with 1.
        0,          # Every step!() call increase step_cur by 1.
        step_max-1, # step_max. The first step is consumed in reset!() call.
        τ_max,      # τ_max
        nothing,    # τ_obj
        nothing,    # V_perm_cur
        nothing,    # V_perm_cycle
        nothing,    # V_perm_obj
        nothing,    # V_disp
        nothing,    # C_perm_cur
        nothing,    # E_total_cur
        nothing,    # E_total_cycle
        nothing,    # mode_cur
        nothing,    # cycle_cur
        nothing,    # last_u
        nothing,    # scenarios
        nothing,    # action
        nothing,    # state
        nothing     # reward
    )

    return SBROEnv
end

