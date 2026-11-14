#!/usr/bin/env bash
# Build the serving image for one profile from upstream TensorFold (the vendor/TensorFold submodule, unmodified) plus
# this repo's patches. Runs on a DGX Spark (aarch64, CUDA toolkit with nvcc for sm_121, Docker); no GPU is used.
#
#   scripts/build.sh tp1|tp2 [STEP]   STEP: source | native | image | all (default all, in that order)
#
#   source  build/PROFILE/src: a clean copy of the submodule at the pinned commit with patches/series.PROFILE applied
#           by `git am`; fails unless the result's tree is the one the series file pins (series.tpN, line `tree`)
#   native  build/PROFILE/out/{bin,lib}: zig build check-native-cuda-host, then tensorfold-native (CUDA sm_121,
#           ReleaseFast, stripped), tf-runner-frontend-check and libtfgrammar (xgrammar v0.2.8); for TP1 also the
#           image decoders the engine pins for --vision, built from their release archives (sha256-checked):
#           libturbojpeg (libjpeg-turbo 3.1.4.1) and libwebpdecoder (libwebp 1.6.0), into lib/
#   image   the Docker layers (docker/): the serving base, the in-model runner layer, and for TP1 the CUDA
#           forward-compatibility layer. TP2's base also carries Mia / MiaAI-Lab's Triton kernel set and the compat
#           driver, copied from her image. Final tag: IMAGE_PREFIX:PROFILE (default qwen4-tensorfold:tp1 / :tp2)
#
# Environment: ZIG (default zig; the engine pins Zig 0.17.0), NVCC (default /usr/local/cuda/bin/nvcc), JOBS (zig -j,
# default 1: the native compile needs about 5 GiB at -j1), XGRAMMAR (an xgrammar checkout at XGRAMMAR_COMMIT with its
# submodules; default build/xgrammar, cloned if missing), MIA_IMAGE (TP2: her recipe's image, default her published
# image by digest), COMPAT_IMAGE (TP1: an image holding the CUDA 13.3 compat driver files; default MIA_IMAGE's base,
# NVIDIA's PyTorch 26.07 container, by digest), IMAGE_PREFIX, DOCKER_BUILD_FLAGS and DOCKER_RUN_FLAGS (extra words for
# each docker build / docker run, e.g. resource limits: DOCKER_BUILD_FLAGS='--memory 4g --memory-swap 4g' with
# DOCKER_BUILDKIT=0, as BuildKit ignores --memory).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
P=${1:-}
STEP=${2:-all}
[[ $P == tp1 || $P == tp2 ]] || { echo "usage: scripts/build.sh tp1|tp2 [source|native|image|all]" >&2; exit 2; }
case $STEP in source | native | image | all) ;; *) echo "unknown step: $STEP" >&2; exit 2 ;; esac
SERIES=patches/series.$P
BASE=$(awk '$1 == "base" {print $2}' "$SERIES")
TREE=$(awk '$1 == "tree" {print $2}' "$SERIES")
W=$ROOT/build/$P
SRC=$W/src
OUT=$W/out
ZIG=${ZIG:-zig}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
JOBS=${JOBS:-1}
XGRAMMAR=${XGRAMMAR:-$ROOT/build/xgrammar}
XGRAMMAR_REPO=https://github.com/mlc-ai/xgrammar.git
XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa # v0.2.8
# The image decoders TP1's --vision loads (the engine refuses any other release: docs/vision.md, JPEG input):
# name, release archive, its sha256.
JPEG_TURBO=(libjpeg-turbo-3.1.4.1 https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
  ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022)
WEBP=(libwebp-1.6.0 https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz
  e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564)
MIA_IMAGE=${MIA_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c}
COMPAT_IMAGE=${COMPAT_IMAGE:-nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}
COMPAT_PATH=/usr/local/cuda-13.3/compat/lib.real
IMAGE_PREFIX=${IMAGE_PREFIX:-qwen4-tensorfold}
TAG=$IMAGE_PREFIX:$P
read -ra BUILD_FLAGS <<< "${DOCKER_BUILD_FLAGS:-}"
read -ra RUN_FLAGS <<< "${DOCKER_RUN_FLAGS:-}"
log() { echo "build.sh $P: $*"; }
die() { echo "build.sh $P: $*" >&2; exit 1; }

manifest() { # DIR -> "sha256  ./path" for every regular file, sorted (the form of the committed manifests)
  (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
}
same_files() { # DIR MANIFEST: DIR holds exactly MANIFEST's files at its sha256s (in any order: sort order is locale's)
  diff <(grep -v '^#' "$2" | grep . | LC_ALL=C sort) <(manifest "$1" | LC_ALL=C sort) > /dev/null || die "$1 differs from $2"
}

step_source() {
  [[ -e $SRC ]] && die "$SRC exists: remove it (or build/$P) for a fresh source tree"
  git submodule update --init vendor/TensorFold
  [[ $(git -C vendor/TensorFold rev-parse HEAD) == "$BASE" ]] || die "vendor/TensorFold is not at $BASE"
  mkdir -p "$W"
  git clone --quiet --no-checkout vendor/TensorFold "$SRC"
  git -C "$SRC" checkout --quiet --detach "$BASE"
  local n=0 p
  while read -r p; do
    [[ -z $p || $p == \#* || $p == base\ * || $p == tree\ * ]] && continue
    git -C "$SRC" -c user.name=qwen4-recipe -c user.email=recipe@example.com -c core.autocrlf=false \
      am --quiet --keep-cr "$ROOT/patches/$p" || die "patch $p did not apply (git -C $SRC am --abort to inspect)"
    n=$((n + 1))
  done < "$SERIES"
  local got
  got=$(git -C "$SRC" rev-parse 'HEAD^{tree}')
  [[ $got == "$TREE" ]] || die "after $n patches the tree is $got, not $TREE (the tree patches/series.$P pins)"
  log "source: $BASE + $n patches = tree $TREE (as patches/series.$P pins)"
}

step_native() {
  [[ -d $SRC ]] || die "no $SRC: run the source step first"
  [[ $(git -C "$SRC" rev-parse 'HEAD^{tree}') == "$TREE" ]] || die "$SRC is not tree $TREE"
  if [[ ! -d $XGRAMMAR ]]; then
    git clone --quiet "$XGRAMMAR_REPO" "$XGRAMMAR"
    git -C "$XGRAMMAR" checkout --quiet --detach "$XGRAMMAR_COMMIT"
    git -C "$XGRAMMAR" submodule update --init --recursive --quiet
  fi
  [[ $(git -C "$XGRAMMAR" rev-parse HEAD) == "$XGRAMMAR_COMMIT" ]] || die "$XGRAMMAR is not at $XGRAMMAR_COMMIT"
  [[ -e $OUT ]] && die "$OUT exists: remove it for a fresh native build"
  # host code for this Spark's own CPU (no -Dtarget / -Dcpu), as the measured images were built
  (cd "$SRC" && "$ZIG" build check-native-cuda-host -j"$JOBS" -Dnvcc="$NVCC" -Dsm=121)
  (cd "$SRC" && "$ZIG" build native -j"$JOBS" -Dnvcc="$NVCC" -Dsm=121 -Doptimize=ReleaseFast -Dstrip=true --prefix "$OUT")
  (cd "$SRC" && "$ZIG" build runner-frontend-check -j"$JOBS" --prefix "$OUT")
  (cd "$SRC" && "$ZIG" build grammar -j"$JOBS" -Dxgrammar="$XGRAMMAR" --prefix "$W/grammar")
  mkdir -p "$OUT/bin" "$OUT/lib"
  local f p
  for f in tensorfold-native tf-runner-frontend-check; do
    p=$(find "$OUT" -type f -name "$f" | head -n 1)
    [[ -x $p ]] || die "no $f in $OUT"
    [[ $p == "$OUT/bin/$f" ]] || cp "$p" "$OUT/bin/"
  done
  p=$(find "$W/grammar" -type f -name libtfgrammar.so | head -n 1)
  [[ -f $p ]] || die "no libtfgrammar.so in $W/grammar"
  cp "$p" "$OUT/lib/"
  (cd "$OUT" && sha256sum bin/tensorfold-native bin/tf-runner-frontend-check lib/libtfgrammar.so)
  [[ $P == tp1 ]] && step_codecs
  return 0
}

codec() { # NAME URL SHA256 CMAKE-ARGS... -> build/codecs/NAME/install: the release archive, checked, built (Release)
  local name=$1 url=$2 sum=$3 dir=$ROOT/build/codecs/$1
  shift 3
  [[ -e $dir ]] && die "$dir exists: remove it for a fresh decoder build"
  mkdir -p "$dir"
  curl -fsSL --retry 3 -o "$dir/src.tar.gz" "$url" || die "could not download $url"
  echo "$sum  $dir/src.tar.gz" | sha256sum -c --quiet - || die "$url: sha256 is not $sum"
  tar -xzf "$dir/src.tar.gz" -C "$dir"
  cmake -S "$dir/$name" -B "$dir/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dir/install" \
    -DCMAKE_INSTALL_LIBDIR=lib "$@" > "$dir/cmake.log" || die "$name: cmake failed ($dir/cmake.log)"
  cmake --build "$dir/build" -j"$JOBS" > "$dir/build.log" || die "$name: build failed ($dir/build.log)"
  cmake --install "$dir/build" > /dev/null || die "$name: install failed"
}

step_codecs() { # TP1's image decoders into $OUT/lib under the file names the engine loads
  codec "${JPEG_TURBO[@]}" -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DWITH_JAVA=OFF -DWITH_TOOLS=OFF -DWITH_TESTS=OFF
  codec "${WEBP[@]}" -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF \
    -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF \
    -DWEBP_BUILD_LIBWEBPMUX=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF
  cp -L "$ROOT/build/codecs/${JPEG_TURBO[0]}/install/lib/libturbojpeg.so.0" "$OUT/lib/"
  cp -L "$ROOT/build/codecs/${WEBP[0]}/install/lib/libwebpdecoder.so.3" "$OUT/lib/"
  grep -aq 'libjpeg-turbo version 3.1.4.1 ' "$OUT/lib/libturbojpeg.so.0" || die "libturbojpeg.so.0 is not 3.1.4.1"
  (cd "$OUT" && sha256sum lib/libturbojpeg.so.0 lib/libwebpdecoder.so.3)
}

copy_out() { # IMAGE PATH DEST: PATH out of IMAGE (pulled if missing)
  docker image inspect "$1" > /dev/null 2>&1 || docker pull "$1"
  local cid
  cid=$(docker create "$1")
  docker cp "$cid:$2" "$3" || { docker rm "$cid" > /dev/null; die "no $2 in $1"; }
  docker rm "$cid" > /dev/null
}

compat_dir() { # IMAGE DEST: the compat driver files and links, exactly as docker/compat pins them
  copy_out "$1" "$COMPAT_PATH" "$2"
  same_files "$2" docker/compat/compat.sha256
  diff <(grep -v '^#' docker/compat/compat.links | LC_ALL=C sort) <(cd "$2" && find . -type l -printf '%p -> %l\n' | LC_ALL=C sort) > /dev/null ||
    die "the links in $1:$COMPAT_PATH differ from docker/compat/compat.links"
  cp docker/compat/tf-compat-entry "$(dirname "$2")/tf-compat-entry"
  chmod 755 "$(dirname "$2")/tf-compat-entry"
}

step_image() {
  [[ -x $OUT/bin/tensorfold-native ]] || die "no $OUT/bin/tensorfold-native: run the native step first"
  local ctx=$W/image
  [[ -e $ctx ]] && die "$ctx exists: remove it for a fresh image build"
  mkdir -p "$ctx/base" "$ctx/runner"
  cp -r "$OUT/bin" "$OUT/lib" "$ctx/base/"
  if [[ $P == tp2 ]]; then
    copy_out "$MIA_IMAGE" /opt/tensorfold/share/tensorfold/cuda/sm121 "$ctx/base/kernels"
    same_files "$ctx/base/kernels" docker/mia-kernels.sha256
    mkdir "$ctx/base/mia-recipe"
    for f in LICENSE.txt NOTICE.txt CREDITS.md; do cp "$SRC/LICENSES/MiaAI-Lab-DualTF-$f" "$ctx/base/mia-recipe/"; done
    compat_dir "$MIA_IMAGE" "$ctx/base/compat"
  fi
  docker build ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} -f "docker/$P.Dockerfile" --build-arg "ENGINE_TREE=$TREE" -t "$TAG-base" "$ctx/base"
  python3 -I scripts/fetch_runner_assets.py config/runner-assets.release config/runner-assets.sha256 "$ctx/runner/runner"
  local runner_tag=$TAG
  [[ $P == tp1 ]] && runner_tag=$TAG-runner
  docker build ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} -f docker/runner.Dockerfile --build-arg "BASE_IMAGE=$TAG-base" -t "$runner_tag" "$ctx/runner"
  if [[ $P == tp1 ]]; then
    mkdir "$ctx/compat"
    compat_dir "$COMPAT_IMAGE" "$ctx/compat/compat"
    docker build ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"} -f docker/compat/Dockerfile --build-arg "BASE_IMAGE=$runner_tag" -t "$TAG" "$ctx/compat"
  fi
  # with no GPU the entry falls back to the host driver, and TF_CUDA_COMPAT=0 says so
  local run=(docker run ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} --rm --network none -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=
    --entrypoint /opt/tensorfold/bin/tf-compat-entry "$TAG" /bin/true)
  [[ $("${run[@]}" 2>&1) == *'host CUDA driver'* ]] || die "$TAG: tf-compat-entry check failed"
  if [[ $P == tp1 ]]; then # --vision's libraries: cuBLASLt found, the JPEG and WebP decoders usable (pinned releases)
    local caps=(docker run ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"} --rm --network none -e NVIDIA_VISIBLE_DEVICES=void
      -e CUDA_VISIBLE_DEVICES= --entrypoint /opt/tensorfold/bin/tensorfold-native "$TAG" capabilities --json)
    "${caps[@]}" > "$W/capabilities.json" 2> "$W/capabilities.err" || die "$TAG: capabilities --json failed ($W/capabilities.err)"
    python3 -I scripts/check_vision_libs.py "$W/capabilities.json" || die "$TAG: --vision libraries ($W/capabilities.json)"
  fi
  log "image: $TAG $(docker image inspect -f '{{.Id}}' "$TAG") (engine tree $TREE)"
}

[[ $STEP == source || $STEP == all ]] && step_source
[[ $STEP == native || $STEP == all ]] && step_native
[[ $STEP == image || $STEP == all ]] && step_image
log "done ($STEP)"
