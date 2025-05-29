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
        u_initial = cfg["u_initial"]
    else
        u_initial = cfg["u_initial"] |> Vector{Float64}
    end

    dt = cfg["dt"] |> Float64

    experience_reset  = reset!(sbro_env, initial_condition, u_initial; dt=dt)

    return json_response(experience_reset)
end

function step_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))

    action = cfg["action"] |> Vector{Float64}

    experience_step  = step!(sbro_env, action)

    return json_response(experience_step)
end

function render_handler(req::HTTP.Request)
    cfg = JSON3.read(String(req.body))

    mode = cfg["mode"] |> Symbol

    rendered_env = render(sbro_env; mode=mode)

    return HTTP.Response(200, rendered_env)
end

function health_check_handler(req::HTTP.Request)
    return HTTP.Response(200, "ok")
end

function cleanup()
    @info "Server termination detected. Cleaning up ..."
    sbro_env = nothing
    return nothing
end

HTTP.register!(ROUTER, "POST", "/reset_scenario", reset_scenario_handler)

HTTP.register!(ROUTER, "POST", "/reset", reset_handler)

HTTP.register!(ROUTER, "POST", "/step", step_handler)

HTTP.register!(ROUTER, "GET", "/render", render_handler)

HTTP.register!(ROUTER, "GET", "/health", health_check_handler)

HTTP.serve(ROUTER, "127.0.0.1", 8081; on_shutdown = cleanup)

# ─── Install signal handlers ─────────────────────────────────────────────────
using Base: Signal, SIGINT, SIGTERM

# Helper to watch one signal
function watch_signal(sig::Int)
    ch = Signal(sig)
    @async while true
        take!(ch)                      # blocks until that OS signal arrives
        @info "Signal $(sig) received—shutting down HTTP server"
        HTTP.close(server)             # start graceful shutdown
        return                         # exit this task
    end
end

watch_signal(SIGINT)                  # Ctrl-C
watch_signal(SIGTERM)                 # `kill` or Docker stop

# ─── Wait for server to finish shutting down ─────────────────────────────────
wait(server)                          # blocks until `close(server)` is done
@info "Server has shut down cleanly."