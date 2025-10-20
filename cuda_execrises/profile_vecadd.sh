#!/bin/bash
# ==========================================
# CUDA Profiling Script for vecAdd
# ==========================================
# Usage:
#   ./profile_vecadd.sh orin
#   ./profile_vecadd.sh rtx
#
# All build and profiling outputs will be stored in ./output/
# ==========================================

set -e  # stop on first error

TARGET=${1:-rtx}   # default = RTX
SRC="vecAdd.cu"
APP="vecAdd"
THREADS=256
PROFILE_TAG=$(date +"%Y%m%d_%H%M")
OUTDIR="output"

mkdir -p "$OUTDIR"

# ---------- Select platform-specific flags ----------
if [ "$TARGET" == "orin" ]; then
    echo "[INFO] Target: Jetson Orin (sm_87)"
    ARCH="sm_87"
    EXTRA_FLAGS="--gpu-architecture=compute_87"
elif [ "$TARGET" == "rtx" ]; then
    echo "[INFO] Target: RTX 3060 (sm_86)"
    ARCH="sm_86"
    EXTRA_FLAGS="--gpu-architecture=compute_86"
else
    echo "[ERROR] Unknown target: $TARGET"
    echo "Usage: ./profile_vecadd.sh [orin|rtx]"
    exit 1
fi

# ---------- Compile ----------
echo "[BUILD] Compiling for $ARCH ..."
nvcc $SRC -o "${OUTDIR}/${APP}_${TARGET}" \
    -std=c++17 -O3 -lineinfo -arch=$ARCH -Xptxas -v $EXTRA_FLAGS

# ---------- Run correctness test ----------
echo "[RUN] Running compute-sanitizer ..."
compute-sanitizer "${OUTDIR}/${APP}_${TARGET}" || echo "[WARN] Compute-sanitizer reported issues."

# ---------- Nsight Systems (timeline) ----------
echo "[PROFILE] Running Nsight Systems ..."
nsys profile -o "${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_sys" \
    --trace=cuda,nvtx,osrt "${OUTDIR}/${APP}_${TARGET}"
nsys stats "${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_sys.nsys-rep" > "${OUTDIR}/${APP}_${TARGET}_sys.txt"

# ---------- Nsight Compute (kernel micro-analysis) ----------
echo "[PROFILE] Running Nsight Compute ..."
ncu --set full --kernel-name "vecAddKernel" \
    --export "${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_ncu" \
    "${OUTDIR}/${APP}_${TARGET}"

# ---------- Summary ----------
echo "[DONE]"
echo
echo "Generated reports under '${OUTDIR}/':"
echo "  - ${APP}_${TARGET}_${PROFILE_TAG}_sys.nsys-rep  (Nsight Systems)"
echo "  - ${APP}_${TARGET}_${PROFILE_TAG}_ncu.ncu-rep (Nsight Compute)"
echo "  - ${APP}_${TARGET}_sys.txt                    (Stats Summary)"
echo
echo "To view reports:"
echo "  # For GUI (requires X11 display):"
echo "  nsys-ui ${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_sys.nsys-rep"
echo "  ncu-ui  ${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_ncu.ncu-rep"
echo
echo "  # For command line analysis:"
echo "  nsys stats ${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_sys.nsys-rep"
echo "  ncu --import ${OUTDIR}/${APP}_${TARGET}_${PROFILE_TAG}_ncu.ncu-rep"
echo