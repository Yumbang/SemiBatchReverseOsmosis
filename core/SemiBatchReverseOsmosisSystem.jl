@mtkmodel SBRO begin
    @description "A dynamic semi-batch reverse osmosis system"

    @constants begin
        m_avg = 44.48828244517122 ,   [description = "Average feed molar mass [g/mol]"]
        k_fp  = 0.6741807585817046e9, [description = "Fouling potential"]
    end

    # Parameters to be assigned
    @parameters begin
        L_m
        W_m
        H_m
        V_pipe_c
        V_pipe_f
        N_modules_per_vessel
        N_vessels
        η_circ_pump
        η_HPP
        Q₀
        mode
        C_feed  # Fresh feed (not mixed)
        T_feed  # Fresh feed (not mixed)
        # P_feed  # Fresh feed (not mixed)
        R_sp
        Rejection
    end

    # Parameters to be calculated
    @parameters begin
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
    end

    
    # Time-dependent variables, unlike parameters (<- time-independent)
    @variables begin
        # Recirculation rate
        α(t)
        # Concentration variables
        C_m_in(t)
        C_m_out(t)
        C_pipe_c_in(t)
        C_pipe_c_out(t)
        C_pipe_f_in(t)
        C_pipe_f_out(t)
        C_disp(t)
        C_perm(t)
        # Flowrate variables
        Q_feed(t)   # Fresh feed Q
        Q_circ(t)   # Circulated brine Q
        Q_disp(t)   # Disposed brine Q
        Q_perm(t)   # Permeate Q
        # Salt mass variables
        M_m(t)
        M_pipe_c(t)
        M_pipe_f(t)
        M_system(t)
        M_feed(t)
        M_disp(t)
        # HRT variables
        τ_feed(t)
        τ_circ(t)
        # Pressure-related variables
        R_m(t)
        π_m_in(t)
        π_m_out(t)
        P_m_in(t)
        P_m_out(t)
        P_circ(t)
        P_disp(t)
        P_circ_pump(t)
        P_HPP(t)
        # Energy-related variables
        Power_circ_pump(t)
        Power_HPP(t)
        SEC(t)  # Ws/m³
    end

    # D() means differential with t
    @equations begin
        # Necessary variables calculation
        α               ~ (1-mode) * (1 / R_sp - 1)
        Q_feed          ~ Q₀       / (1 + α)
        Q_circ          ~ α * Q₀   / (1 + α)
        Q_disp          ~ ( mode ) * Q₀ * (1 - R_sp)
        Q_perm          ~ Q₀ * R_sp
        
        # HRT variables (hr)
        τ_feed          ~ V_pipe_f / Q_feed
        τ_circ          ~ V_pipe_c / Q_circ # Becomes Inf in purge mode. Don't worry, it's intended.
        
        # Core RO dynamics.
        # Circulation pipe dynamics
        C_pipe_c_in     ~ C_m_out
        D(C_pipe_c_out) ~ (1-mode) * (C_pipe_c_in - C_pipe_c_out) / τ_circ / 3600
        # Feed pipe dynamics
        C_pipe_f_in     ~ (1-mode) * (C_feed  + α * C_pipe_c_out) / (1 + α) +
                          ( mode ) * (C_feed)
        D(C_pipe_f_out) ~ (C_pipe_f_in - C_pipe_f_out) / τ_feed / 3600                 # Mode independent behavior
        # System salt mass balance dynamics
        C_m_in          ~ (C_pipe_f_out)
        C_perm          ~ Cp * (C_m_in + C_m_out) / 2 * (1 - Rejection)
        Q_feed * C_feed -
        Q_perm * C_perm ~  D(C_m_in)  * 3600 * (V_pipe_f + V_m / 2) + D(C_m_out) * 3600 * (V_m / 2) +
                           (1-mode) * (
                             D(C_pipe_c_out) * V_pipe_c * 3600
                           ) +
                           ( mode ) * (
                             C_m_out * Q_disp
                           )
        # Fouling dynamics
        D(R_m)          ~ k_fp * J_w


        # Observable variables
        C_disp          ~ ( mode ) * C_m_out
        M_m             ~ V_m * (C_m_in + C_m_out) / 2
        M_pipe_c        ~ V_pipe_c * (C_pipe_c_out)
        M_pipe_f        ~ V_pipe_f * (C_pipe_f_out)
        M_system        ~ M_m + M_pipe_c + M_pipe_f
        π_m_in          ~ 2 / m_avg * C_m_in  * 8.3145e3 * (T_feed + 273.15)     # Assuming the ions are Na⁺
        π_m_out         ~ 2 / m_avg * C_m_out * 8.3145e3 * (T_feed + 273.15)     # Assuming the ions are Na⁺
        P_m_in          ~ Cp * (π_m_in + π_m_out) / 2 + (J_w * R_m) + ΔP_m / 2
        P_m_out         ~ P_m_in - ΔP_m
        P_circ          ~ (1-mode) * P_m_out
        P_disp          ~ ( mode ) * P_m_out
        P_circ_pump     ~ 2 * ΔP_m  # Assuming that pressure loss during recirculation is approximately equal to that during membrane filtration. Need to be improved later.
        P_HPP           ~ P_m_in    # - P_feed
        Power_circ_pump ~ P_circ_pump * Q_circ / 3600 / η_circ_pump
        Power_HPP       ~ P_HPP       * Q_feed / 3600 / η_HPP
        SEC             ~ (Power_circ_pump + Power_HPP) / (Q_perm / 3600)
    end
end