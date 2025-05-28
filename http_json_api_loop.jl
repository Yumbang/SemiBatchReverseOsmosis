using HTTP, JSON3
include("rl_plans/plan_A.jl")

function json_response(payload; status::Integer = 200)
    return HTTP.Response(
        status,
        ["Content-Type" => "application/json"],
        JSON3.write(payload),
    )
end

print("🚀 Starting SBRO environment ... ")
sbro_env = initialize_sbro_env(
    ; dt = 30.0, τ_max=86400.0
)
println("Done!")

const ROUTER = HTTP.Router()

function reset_scenario_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))
    
    scenario_condition  = cfg["scenario_condition"]  |> Vector{Float64}
    objective_condition = cfg["objective_condition"] |> Vector{Float64}

    reset_scenario!(sbro_env, scenario_condition, objective_condition)

    return HTTP.Response(200, "Scenario set")
end

function reset_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))
    
    initial_condition = cfg["action"]    |> Vector{Float64}

    if isnothing(cfg["u_initial"]) 
        u_initial         = cfg["u_initial"]
    else
        u_initial         = cfg["u_initial"] |> Vector{Float64}
    end

    dt                = cfg["dt"] |> Float64

    experience_reset  = reset!(sbro_env, initial_condition, u_initial; dt=dt)

    return json_response(experience_reset)
end

function step_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))

    action = cfg["action"]  |> Vector{Float64}

    experience_step  = step!(sbro_env, action)

    return json_response(experience_step)
end

function render_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))

    mode = cfg["mode"]  |> Symbol

    rendered_env  = render(sbro_env; mode=mode)

    return HTTP.Response(200, rendered_env)
end

HTTP.register!(ROUTER, "POST", "/reset_scenario", reset_scenario_handler)

HTTP.register!(ROUTER, "POST", "/reset", reset_handler)

HTTP.register!(ROUTER, "POST", "/step", step_handler)

HTTP.register!(ROUTER, "GET", "/render", render_handler)

HTTP.serve(ROUTER, "127.0.0.1", 8081)