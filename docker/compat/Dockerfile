# The CUDA 13.3 forward-compatibility layer over the TP1 runner image (scripts/build.sh tp1 image): the user-mode driver
# files (libcuda 610.43.02) as Mia / MiaAI-Lab's NGC-based image ships them in /usr/local/cuda-13.3/compat/lib.real,
# checked against compat.sha256 and compat.links, in /opt/tensorfold/compat, and the entrypoint tf-compat-entry. The
# entry puts them first on the loader path only while the host kernel module is older than 610 and cuInit succeeds
# through them; otherwise the host driver. TF_CUDA_COMPAT=0 forces the host driver. No engine file changes.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY compat/ /opt/tensorfold/compat/
COPY tf-compat-entry /opt/tensorfold/bin/tf-compat-entry
RUN test -f /opt/tensorfold/compat/libcuda.so.610.43.02 && test -e /opt/tensorfold/compat/libcuda.so.1 \
    && test -x /opt/tensorfold/bin/tf-compat-entry && sh -n /opt/tensorfold/bin/tf-compat-entry
LABEL org.tensorfold.cuda-compat="610.43.02 (Mia / MiaAI-Lab image, cuda-13.3/compat/lib.real)"
ENTRYPOINT ["tf-compat-entry", "tensorfold-native"]
