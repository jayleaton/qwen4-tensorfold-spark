# The in-model runner layer over a serving base (scripts/build.sh): clang-18 / lld-18 (the runner's RV32 frontend), the
# circuit calc.tfpc and the graph-v7-powmod bank (runner/, fetched by scripts/fetch_runner_assets.py and checked against
# config/runner-assets.sha256). The build fails unless tf-runner-frontend-check computes the engine's probe cells.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18 \
    && ln -sf clang-18 /usr/bin/clang && ln -sf ld.lld-18 /usr/bin/ld.lld && rm -rf /var/lib/apt/lists/*
COPY runner/calc.tfpc /opt/tensorfold/share/runner/calc.tfpc
COPY runner/graph-v7-powmod/bank/ /opt/tensorfold/share/runner/graph-v7-powmod/bank/
COPY runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/licenses/runtime-deps/Qwenputer-MIT.txt
RUN /opt/tensorfold/bin/tf-runner-frontend-check --require --circuit /opt/tensorfold/share/runner/calc.tfpc
LABEL org.tensorfold.runner.circuit="calc.tfpc" org.tensorfold.runner.graph="graph-v7-powmod" \
      org.tensorfold.runner.assets="runner-assets-20261107 b5c40d32bcab"
