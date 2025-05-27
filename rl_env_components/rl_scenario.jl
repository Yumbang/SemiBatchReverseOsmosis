"""
Generate a fresh feed water temperature scenario.
The temperature follows N(μ_T, σ_T²).
"""
function generate_T_feed_scenario(μ_T, σ_T; dt, len_episode, rng=Random.GLOBAL_RNG)
    @assert dt > 0 && len_episode > 0 "dt and len_episode must be positive"

    # exact integer step count
    n_steps = ceil(Int, len_episode / dt)

    # μ_T + Gaussian noise
    temperature = rand(rng, Normal(μ_T, σ_T), n_steps)

    return temperature
end

"""
Generate a fresh feed water temperature scenario.
The temperature follows periodic sin-shaped distribution with N(0, σ_T²) noise.
"""
function generate_T_feed_scenario(low_T, high_T, σ_T; dt, len_episode, rng=Random.GLOBAL_RNG)
    @assert dt > 0 && len_episode > 0 "dt and len_episode must be positive"

    # exact integer step count
    n_steps = ceil(Int, len_episode / dt)

    # Assume 24-hour, sin-shaped periodic temperature with random starting time
    starting_time = 2π * rand(rng)
    temperature   = sin.(
        range(0, n_steps-1) .* (dt * 2π / (60*60*24)) .+ starting_time
    ) ./ (2 / (high_T - low_T)) .+ (high_T + low_T) / 2

    # Gaussian noise
    noise = rand(rng, Normal(0.0, σ_T), n_steps)

    return temperature + noise
end

"""
Generate a fresh feed water concentration scenario.
The concentration follows N(μ_C, σ_C²).
"""
function generate_C_feed_scenario(μ_C, σ_C; dt, len_episode, rng=Random.GLOBAL_RNG)
    @assert dt > 0 && len_episode > 0 "dt and len_episode must be positive"

    # exact integer step count
    n_steps = ceil(Int, len_episode / dt)

    # μ_C + Gaussian noise
    concentration = rand(rng, Normal(μ_C, σ_C), n_steps)

    return concentration
end

function generate_objectives_scenario(low_τ_obj, high_τ_obj, low_V_perm_obj, high_V_perm_obj, rng=Random.GLOBAL_RNG)
    τ_obj      = rand(rng) * (high_τ_obj      - low_τ_obj)      + low_τ_obj
    V_perm_obj = rand(rng) * (high_V_perm_obj - low_V_perm_obj) + low_V_perm_obj
    return (τ_obj, V_perm_obj)
end