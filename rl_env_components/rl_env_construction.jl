mutable struct SemiBatchReverseOsmosisEnv
    const sys   :: ODESystem          # Semi-batch RO system (simplified)
    problem     :: ODEProblem         # Semi-batch RO ODE problem
    setp_handle :: SymbolicIndexingInterface.ParameterHookWrapper # Parameter updater
    dt          :: Float64            # Δt for each environment step [s]. NOT THAT OF ODE!
    episode     :: Int                # episode counter
end

"""
Initialize a new SemiBatchReverseOsmosisEnv.
None of the `numbers` (i.e., C_feed, T_feed, Q₀ etc.), except dt, are not designed to be used.
To configure actual sbro ODE problem, use reset() in desired [planX.jl] file.

dt: Δt for each environment step [s]
"""
function initialize_sbro_env(;dt::Float64)
    dummy_C_feed = 0.05
    dummy_T_feed = 15.0

    simple_sbro  = make_simple_sbro_system();
    def_param    = default_parameters(simple_sbro; C_feed = dummy_C_feed, T_feed = dummy_T_feed);
    sbro_prob,
     setp_handle = make_sbro_problem(simple_sbro, def_param...; dt = 0.0);   # Dummy problem, tspan with no length

    SBROEnv = SemiBatchReverseOsmosisEnv(
        simple_sbro,
        sbro_prob,
        setp_handle,
        dt,
        0   # Every reset() call increase episode by 1. I.e., episode must start with 1.
    );

    return SBROEnv
end

