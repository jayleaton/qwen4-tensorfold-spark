# TP1 serving base (scripts/build.sh tp1 image): the engine scripts/build.sh built from vendor/TensorFold plus
# patches/series.tp1 (bin/ and lib/ in the build context), on NVIDIA's CUDA 13.0.1 base with NCCL, ibverbs and python3
# (the compute lane's interpreter), and what --vision loads: cuBLASLt (the image tower's GEMMs, NVIDIA's libcublas-13-0)
# and the two image decoders the engine pins, libturbojpeg (libjpeg-turbo 3.1.4.1) and libwebpdecoder (libwebp 1.6.0),
# which scripts/build.sh built into lib/ (the engine looks in ../lib from its bin/). No compile happens in Docker. The
# runner layer (runner.Dockerfile) and the CUDA forward-compatibility layer (compat/Dockerfile) go over it.
FROM nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers ca-certificates \
    python3 libcublas-13-0 && rm -rf /var/lib/apt/lists/*
COPY bin/ /opt/tensorfold/bin/
COPY lib/ /opt/tensorfold/lib/
# every NEEDED resolves; NCCL, ibverbs and cuBLASLt are on the loader path; python3 is where the lane looks
# (TF_COMPUTE_LANE_PYTHON); both decoders resolve too
RUN ! ldd /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/lib/libtfgrammar.so /opt/tensorfold/lib/libturbojpeg.so.0 \
    /opt/tensorfold/lib/libwebpdecoder.so.3 | grep "not found" \
    && ldconfig -p | grep -q "libnccl.so.2 " && ldconfig -p | grep -q "libibverbs.so.1 " \
    && ldconfig -p | grep -q "libcublasLt.so.13 " && test -x /usr/bin/python3
ENV PATH=/opt/tensorfold/bin:$PATH
ARG ENGINE_TREE=""
LABEL org.tensorfold.engine-tree="${ENGINE_TREE}" org.tensorfold.base="ashhart/TensorFold aaac299d1 + patches/series.tp1"
ENTRYPOINT ["tensorfold-native"]
