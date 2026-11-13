# TP2 serving base (scripts/build.sh tp2 image): the engine scripts/build.sh built from vendor/TensorFold plus
# patches/series.tp2 (bin/ and lib/), Mia / MiaAI-Lab's captured Triton kernel set (kernels/, copied byte for byte from
# her recipe's image and checked against docker/mia-kernels.sha256), her LICENSE, NOTICE and CREDITS (mia-recipe/), and
# the CUDA 13.3 forward-compatibility driver her image ships (compat/, checked against docker/compat/compat.sha256)
# with its entrypoint tf-compat-entry. No compile happens in Docker. The runner layer (runner.Dockerfile) goes over it.
FROM nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers ca-certificates \
    python3 && rm -rf /var/lib/apt/lists/*
COPY bin/ /opt/tensorfold/bin/
COPY lib/ /opt/tensorfold/lib/
COPY kernels/ /opt/tensorfold/share/tensorfold/cuda/sm121/
COPY mia-recipe/ /opt/mia-recipe/
COPY compat/ /opt/tensorfold/compat/
COPY tf-compat-entry /opt/tensorfold/bin/tf-compat-entry
# every NEEDED resolves; NCCL and ibverbs are on the loader path; python3 is where the lane looks (TF_COMPUTE_LANE_PYTHON)
RUN ! ldd /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/lib/libtfgrammar.so | grep "not found" \
    && ldconfig -p | grep -q "libnccl.so.2 " && ldconfig -p | grep -q "libibverbs.so.1 " && test -x /usr/bin/python3 \
    && test -f /opt/tensorfold/compat/libcuda.so.610.43.02 && test -x /opt/tensorfold/bin/tf-compat-entry
ENV PATH=/opt/tensorfold/bin:$PATH TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 CUDA_CACHE_PATH=/cache/nv
ARG ENGINE_TREE=""
LABEL org.tensorfold.engine-tree="${ENGINE_TREE}" org.tensorfold.base="ashhart/TensorFold aaac299d1 + patches/series.tp2" \
      org.tensorfold.cuda-compat="610.43.02 (Mia / MiaAI-Lab image)" \
      org.tensorfold.credit="engine and Triton kernel set: Mia / MiaAI-Lab dual-Spark recipe dc05c912 (Apache-2.0), /opt/mia-recipe"
ENTRYPOINT ["tf-compat-entry", "tensorfold-native"]
