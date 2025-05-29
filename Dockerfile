FROM julia:1.11.5-alpine AS build

WORKDIR /app
COPY Manifest.toml Project.toml ./
ENV JULIA_PROJECT=/app
RUN julia -e 'using Pkg; Pkg.instantiate()'

# keep directory structure intact ↓↓↓
COPY core              /app/core
COPY rl_env_components /app/rl_env_components
COPY rl_plans          /app/rl_plans
COPY http_json_api_loop.jl /app/

RUN julia -e 'using Pkg; include("rl_plans/plan_A.jl"); Pkg.precompile(); println("🛠️  Precompilation done")'

# ─── runtime stage ───────────────────────────────────────────────────────────
FROM julia:1.11.5-alpine
WORKDIR /app
COPY --from=build /usr/local/julia /usr/local/julia
COPY --from=build /app /app

ENV JULIA_DEPOT_PATH="/usr/local/julia"
EXPOSE 8081
CMD ["julia", "http_json_api_loop.jl"]
