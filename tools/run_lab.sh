#!/usr/bin/env bash
set -e

LAB="${1:-work1}"

if [ ! -d "${LAB}" ]; then
  echo "❌ Error: directory '${LAB}' not found in repository root."
  echo "Usage: ./tools/run_lab.sh [work1|work2|work3|...]"
  exit 1
fi

RESULTS_DIR="${LAB}/results"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo " 1. Building project (${LAB})"
echo "=========================================="
ninja -C build

echo ""
echo "=========================================="
echo " 2. Running unit tests"
echo "=========================================="
"./build/${LAB}/tests/hsys_tests"

echo ""
echo "=========================================="
echo " 3. Running benchmarks"
echo "=========================================="
"./build/${LAB}/benchmarks/hsys_benchmarks" \
  --benchmark_out="${RESULTS_DIR}/benchmark_results.json" \
  --benchmark_out_format=json

echo ""
echo "=========================================="
echo " 4. Generating Plotly charts"
echo "=========================================="
python3 tools/benchmark-plot/complexity_chart.py \
  -j "${RESULTS_DIR}/benchmark_results.json" \
  -c "${RESULTS_DIR}/complexity_chart.html" \
  --xlog --ylog

if [ "${LAB}" = "work3" ]; then
  TARGET="BM_CudaMatrixMulShmem"
  REF="BM_CudaMatrixMulNaive"
else
  TARGET=$(python3 -c "import json; data=json.load(open('${RESULTS_DIR}/benchmark_results.json')); print([b['name'].split('/')[0] for b in data['benchmarks'] if 'cuda' in b['name'].lower()][0])")
  REF=$(python3 -c "import json; data=json.load(open('${RESULTS_DIR}/benchmark_results.json')); print([b['name'].split('/')[0] for b in data['benchmarks'] if 'eigen' in b['name'].lower() or 'cpu' in b['name'].lower()][0])")
fi

python3 tools/benchmark-plot/speedup_chart.py \
  -j "${RESULTS_DIR}/benchmark_results.json" \
  -t "${TARGET}" \
  -r "${REF}" \
  -c "${RESULTS_DIR}/speedup_chart.html" \
  --baseline --xlog

echo ""
echo "=========================================="
echo "   All results saved to: ${RESULTS_DIR}"
echo "   • ${RESULTS_DIR}/benchmark_results.json"
echo "   • ${RESULTS_DIR}/complexity_chart.html"
echo "   • ${RESULTS_DIR}/speedup_chart.html"
echo "=========================================="
