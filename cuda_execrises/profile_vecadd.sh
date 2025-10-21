#!/bin/bash
# ==========================================
# CUDA Profiling Script for Multiple Programs
# ==========================================
# Usage:
#   ./profile_vecadd.sh vecAdd orin
#   ./profile_vecadd.sh grayscale rtx
#   ./profile_vecadd.sh vecAdd rtx    (default)
#
# All build and profiling outputs will be stored in ./output/
# ==========================================

set -e  # stop on first error

APP=${1:-vecAdd}   # default = vecAdd
TARGET=${2:-rtx}    # default = RTX
PROFILE_TAG=$(date +"%Y%m%d_%H%M")
OUTDIR="output"

# ---------- Select source file based on app ----------
if [ "$APP" == "vecAdd" ]; then
    SRC="vecAdd.cu"
    KERNEL_NAME="vecAddKernel"
elif [ "$APP" == "grayscale" ]; then
    SRC="grayscale.cu"
    KERNEL_NAME="grayscaleKernel"
else
    echo "[ERROR] Unknown app: $APP"
    echo "Usage: ./profile_vecadd.sh [vecAdd|grayscale] [orin|rtx]"
    exit 1
fi

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
    echo "Usage: ./profile_vecadd.sh [vecAdd|grayscale] [orin|rtx]"
    exit 1
fi

# ---------- Compile ----------
echo "[BUILD] Compiling $SRC for $ARCH ..."
if [ "$APP" == "grayscale" ]; then
    # For grayscale, STB headers are now in the same directory
    nvcc $SRC -o "${OUTDIR}/${APP}_${TARGET}" \
        -std=c++17 -O3 -lineinfo -arch=$ARCH -Xptxas -v $EXTRA_FLAGS \
        -lcudart
else
    nvcc $SRC -o "${OUTDIR}/${APP}_${TARGET}" \
        -std=c++17 -O3 -lineinfo -arch=$ARCH -Xptxas -v $EXTRA_FLAGS
fi

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
ncu --set full --kernel-name "$KERNEL_NAME" \
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