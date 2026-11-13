# TP1 serving base (scripts/build.sh tp1 image): the engine scripts/build.sh built from vendor/TensorFold plus
# patches/series.tp1 (bin/ and lib/ in the build context), on NVIDIA's CUDA 13.0.1 base with NCCL, ibverbs and python3
# (the compute lane's interpreter). No compile happens in Docker. The runner layer (runner.Dockerfile) and the CUDA
# forward-compatibility layer (compat/Dockerfile) go over it.
FROM nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers ca-certificates \
    python3 && rm -rf /var/lib/apt/lists/*
COPY bin/ /opt/tensorfold/bin/
COPY lib/ /opt/tensorfold/lib/
# every NEEDED resolves; NCCL and ibverbs are on the loader path; python3 is where the lane looks (TF_COMPUTE_LANE_PYTHON)
RUN ! ldd /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/lib/libtfgrammar.so | grep "not found" \
    && ldconfig -p | grep -q "libnccl.so.2 " && ldconfig -p | grep -q "libibverbs.so.1 " && test -x /usr/bin/python3
ENV PATH=/opt/tensorfold/bin:$PATH
ARG ENGINE_TREE=""
LABEL org.tensorfold.engine-tree="${ENGINE_TREE}" org.tensorfold.base="ashhart/TensorFold f212eeabd + patches/series.tp1"
ENTRYPOINT ["tensorfold-native"]
