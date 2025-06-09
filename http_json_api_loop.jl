using HTTP, JSON3, Dates
include("rl_plans/plan_A.jl")

function json_response(payload; status::Integer = 200)
    return HTTP.Response(
        status,
        ["Content-Type" => "application/json"],
        JSON3.write(payload),
    )
end

print("[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] 🚀 Starting SBRO environment ... ")
sbro_env = initialize_sbro_env(
    ; dt = 30.0, τ_max=86400.0
)
println("Done!")

const ROUTER = HTTP.Router()

function reset_scenario_handler(req::HTTP.Request)
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Scenario reset requested."
    cfg = JSON3.read(String(req.body))
    
    scenario_condition  = cfg["scenario_condition"]  |> Vector{Float64}
    objective_condition = cfg["objective_condition"] |> Vector{Float64}

    reset_scenario!(sbro_env, scenario_condition, objective_condition)

    return HTTP.Response(200, "Scenario set")
end

function reset_handler(req::HTTP.Request)
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Environment reset requested."
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

function hard_reset_handler(req::HTTP.Request)
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Environment hard reset requested"

    cfg = JSON3.read(String(req.body))

    if isnothing(cfg["dt"])
        dt = 30.0
    else
        dt = cfg["dt"]
    end

    if isnothing(cfg["time_max"])
        τ_max = 86400.0
    else
        τ_max = cfg["time_max"]
    end

    hard_reset!(sbro_env; dt = dt, τ_max = τ_max)

    return HTTP.Response(200, "Environment hard reset.")
end

function step_handler(req::HTTP.Request)
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Environment step requested <STEP $(sbro_env.step_cur) ▶ $(sbro_env.step_cur+1)>."
    cfg = JSON3.read(String(req.body))

    action = cfg["action"] |> Vector{Float64}

    experience_step  = step!(sbro_env, action)

    return json_response(experience_step)
end

function render_handler(req::HTTP.Request)
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Environment render requested."

    rendered_env = render(sbro_env; mode=:text)

    return HTTP.Response(200, rendered_env)
end

function health_check_handler(req::HTTP.Request)
    return HTTP.Response(200, "ok")
end

function cleanup()
    @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Server termination detected. Cleaning up ..."
    sbro_env = nothing
    return nothing
end

HTTP.register!(ROUTER, "POST", "/reset_scenario", reset_scenario_handler)

HTTP.register!(ROUTER, "POST", "/hard_reset", hard_reset_handler)

HTTP.register!(ROUTER, "POST", "/reset", reset_handler)

HTTP.register!(ROUTER, "POST", "/step", step_handler)

HTTP.register!(ROUTER, "GET", "/render", render_handler)

HTTP.register!(ROUTER, "GET", "/health", health_check_handler)

HTTP.serve(ROUTER, "0.0.0.0", 8081; on_shutdown = cleanup)

# ─── Install signal handlers ─────────────────────────────────────────────────
using Base: Signal, SIGINT, SIGTERM

# Helper to watch one signal
function watch_signal(sig::Int)
    ch = Signal(sig)
    @async while true
        take!(ch)                      # blocks until that OS signal arrives
        @info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Signal $(sig) received—shutting down HTTP server"
        HTTP.close(server)             # start graceful shutdown
        return                         # exit this task
    end
end

watch_signal(SIGINT)                  # Ctrl-C
watch_signal(SIGTERM)                 # `kill` or Docker stop

# ─── Wait for server to finish shutting down ─────────────────────────────────
wait(server)                          # blocks until `close(server)` is done
@info "[$(Dates.format(now(), "yy.mm.dd/HH:MM:SS"))] Server has shut down cleanly."