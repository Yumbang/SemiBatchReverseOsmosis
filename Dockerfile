# Maybe not the most beautiful Dockerfile possible, but I'm Noob in Docker now.
# TY LLM 🤖
# Note 1: This Dockerfile works, but as you start a new instance of this image,
#        it takes forever to precompile the packages.

###############################################################################
# 1. build stage: resolve and precompile dependencies
###############################################################################
FROM julia:1.11.5-bookworm AS build

WORKDIR /app
ENV JULIA_PROJECT=/app

COPY Manifest.toml Project.toml ./
RUN julia -e 'using Pkg; Pkg.instantiate()'

# your source code
COPY core              /app/core
COPY rl_env_components /app/rl_env_components
COPY rl_plans          /app/rl_plans
COPY http_json_api_loop.jl /app/

RUN julia -e 'using Pkg; include("rl_plans/plan_A.jl"); Pkg.precompile(); println("🛠️  Precompilation done")'

###############################################################################
# 2. runtime stage: only Julia + depot + your code
###############################################################################
FROM julia:1.11.5-bookworm

# copy Julia itself
COPY --from=build /usr/local/julia /usr/local/julia

# copy the fully-prepared depot
COPY --from=build /root/.julia /usr/local/share/julia
ENV  JULIA_DEPOT_PATH=/usr/local/share/julia

# copy your project
WORKDIR /app
COPY --from=build /app /app
ENV  JULIA_PROJECT=/app

EXPOSE 8081
CMD ["julia", "http_json_api_loop.jl"]
