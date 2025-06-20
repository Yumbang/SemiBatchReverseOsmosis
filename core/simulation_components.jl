"""
Initialize a simplified semi-batch RO system
"""
function make_simple_sbro_system()
    @named semibatch_ro = SBRO()
    simple_sbro = structural_simplify(semibatch_ro)
    return simple_sbro
end

"""
Generate default parameter and initial variables for ODE problem building.
Most of the values are from the membrane data sheet (FILMTEC™ BW30-4040).

C_feed: Concentration of fresh feed [kg/m³]
T_feed: Temperature of fresh feed [°C]
"""
function default_parameters(sys; C_feed, T_feed, Q₀=5.0, R_sp=0.5)
    A_f    = π * (40.0u"mm"/2)^2 |> u"m^2" |> ustrip
    L_f    = 4.8u"m" |> ustrip
    V_f    = A_f * L_f
    A_c    = π * (25.0u"mm"/2)^2 |> u"m^2" |> ustrip
    L_c    = 6.0u"m" |> ustrip
    V_c    = A_c * L_c

    spec_membrane_channel_H = 34.0u"mil"   |> u"m"   |> ustrip
    spec_membrane_Area      = 78.0u"ft^2"  |> u"m^2" |> ustrip
    spec_membrane_Len       = 40.0u"inch"  |> u"m"   |> ustrip
    spec_membrane_Width     = spec_membrane_Area / spec_membrane_Len

    spec_membrane_Perm      = 3.790745450699585u"L/bar/hr/m^2" |> u"m/Pa/s"
    spec_membrane_R         = ustrip(1 / spec_membrane_Perm)

    # Process configuration parameters are hard-coded (e.g., N_modules_per_vessel, N_vessels)
    sbro_param = [
        sys.L_m         => spec_membrane_Len,
        sys.W_m         => spec_membrane_Width,
        sys.H_m         => spec_membrane_channel_H,
        sys.V_pipe_c    => V_c,
        sys.V_pipe_f    => V_f,
        sys.N_modules_per_vessel => 2,
        sys.N_vessels   => 4,
        sys.η_circ_pump => 0.4,
        sys.η_HPP       => 0.8,
        sys.Q₀          => Q₀,
        sys.mode        => 0,   # Start by CC mode, by default
        sys.C_feed      => C_feed,
        sys.T_feed      => T_feed,
        sys.R_sp        => R_sp,
        sys.Rejection   => 0.995
    ]

    sbro_vars = [
        C_feed,                 # C_pipe_f_out(t)
        C_feed,                 # C_pipe_c_out(t)
        C_feed,                 # C_m_out(t)
        spec_membrane_R / 1.5   # R_m(t)
    ]

    return (sbro_param, sbro_vars)
end


"""
Initialize a semi-batch RO differential equation problem, starting from t = 0.0 s

sys: SBRO system (original or *simplified* << RECOMMENDED!)
system_parameters: SBRO system parameters map
initial_variables: SBRO ODE problem's initial variables vector
"""
function make_sbro_problem(sys, system_parameters, initial_variables; dt)
    tspan = (0.0, dt)
    sbro_problem = ODEProblem(sys, initial_variables, tspan, system_parameters)
    setp_handle = setp(sys,[sys.Q₀
                            sys.mode
                            sys.C_feed
                            sys.T_feed
                            sys.R_sp
                            sys.μ
                            sys.ν
                            sys.Dif
                            sys.u_in
                            sys.u_out
                            sys.Sc
                            sys.Re
                            sys.Sh
                            sys.K
                            sys.J_w
                            sys.Cp
                            sys.ΔP_m])

    return (sbro_problem, setp_handle)
end

"""
Function for efficient parameter update
par: Current problem's tunable parameters
new_op_par: New operation-related parameters (Q₀, mode, C_feed, T_feed, R_sp)
"""
function calculate_new_para(par, new_op_par)
    # The following parameter order is hard-coded.
    # If SBRO system's parameter configuration changes, the order must be revised.
    Sc          = par[3]
    μ           = par[4]
    Sh          = par[5]
    dh          = par[6]
    ν           = par[7]
    N_vessels   = par[8]
    V_m         = par[10]
    f_m         = par[11]
    K           = par[13]
    L_m         = par[14]
    u_in        = par[16] 
    u_out       = par[17]
    W_m         = par[18]
    Cp          = par[19]
    Re          = par[21]
    J_w         = par[22]
    Dif         = par[25]
    ΔP_m        = par[26]
    N_modules_per_vessel = par[28]
    H_m         = par[30]
    
    Q₀          = new_op_par[1]
    mode        = new_op_par[2]
    C_feed      = new_op_par[3]
    T_feed      = new_op_par[4]
    R_sp        = new_op_par[5]

    V_m   = L_m * W_m * H_m * N_modules_per_vessel * N_vessels
    μ     = 2.414e-5 * 10^(247.8 / (T_feed + 273.15 - 140))       # Dynamic viscosity from Prof. KH Jeong's model
    ν     = μ / 1e3                                               # Kinematic viscosity
    dh    = 2 * W_m * H_m / (W_m + H_m)                           # Hydraulic diameter [m]
    # dh    = H_m / 2                                               # Hydraulic diameter from Prof. KH Park's model [m]
    Dif   = 6.725e-6 * exp(1.546e-4 * C_feed - 2513 / (T_feed + 273.15))     # Diffusivity [m²/s]
    u_in  =              Q₀ / (W_m * H_m * N_vessels) / 3600      # Flow velocity at membrane feed (mixed feed)
    u_out = (1 - R_sp) * Q₀ / (W_m * H_m * N_vessels) / 3600      # Flow velocity at membrane exit (brine)
    Sc    = ν / Dif
    Re    = dh * (u_in + u_out) / 2 / ν
    Sh    = 0.065 * Re^0.875 * Sc^0.25                            # Sherwood number from Prof. KH Jeong's model
    # Sh    = 0.2 * Re^0.57 * Sc^0.40                               # Sherwood number from Prof. KH Park's model
    K     = Sh * Dif / dh                                         # Mass transfer coefficient [m/s]
    J_w   = R_sp * Q₀ / 3600 / (L_m * W_m * N_modules_per_vessel * N_vessels)  # Average transmembrane flux
    Cp    = exp(J_w / K)                                          # Concentration polarization factor
    f_m   = 20.0                                                  # Friction factor
    ΔP_m  = (12 * 32.0 * μ * ((u_in + u_out) / 2) * L_m * N_modules_per_vessel) / H_m^2    # Pressure loss from Prof. KH Jeong's model
    # ΔP_m  = f_m * μ * ((u_in + u_out) / 2) * L_m * N_modules_per_vessel / dh^2

    # Return vector's order must follow that used in setp_handle initialization
    return [
        Q₀
        mode
        C_feed
        T_feed
        R_sp
        μ
        ν
        Dif
        u_in
        u_out
        Sc
        Re
        Sh
        K
        J_w
        Cp
        ΔP_m
    ]
end

function update_parameters(sys, problem, setp_handle, new_op_var, last_u; dt, reset_tspan=false)
    sbro_var        = deepcopy(last_u)
    sbro_param      = parameter_values(problem)
    sbro_param_new  = calculate_new_para(sbro_param.tunable, new_op_var)

    if reset_tspan
        tspan_new       = (0.0, dt)
    else
        tspan_old       = problem.tspan
        tspan_new       = (tspan_old[2], tspan_old[2]+dt)
    end

    setp_handle(sbro_param, sbro_param_new)
    problem.ps[Initial.(unknowns(sys))] = sbro_var

    problem = remake(problem; p = sbro_param, tspan = tspan_new)

    return problem
end