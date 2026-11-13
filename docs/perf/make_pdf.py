import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

R = "benchmarks/results/cpu_perf/"
b = json.load(open(R + "baseline_run2.json"))
b1 = json.load(open(R + "baseline_run1.json"))
a = json.load(open(R + "perf_cpu.json"))
o = json.load(open(R + "perf_cpu_1thread.json"))
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10)
H1, H2 = ss["Title"], ss["Heading2"]
st = []
P = lambda t, s=body: st.append(Paragraph(t, s))
def tbl(data, widths, fs=7.5, right=True):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), fs), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f7ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT" if right else "LEFT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")])]))
    st.append(t)

P("MLX CPU backend: measured speedups on the perf/cpu branch", H1)
P("Fork: github.com/shubhxho/mlx, branch <b>perf/cpu</b>. Baseline: upstream ml-explore/mlx commit 96fbf67. "
  "Every number in this document was measured. None are projected or estimated.")
P("Summary", H2)
P("Ten commits make the CPU backend multithreaded for elementwise ops, copies, reductions, softmax and sort. "
  "They also make the polynomial exp vectorizable with GCC on x86, use multiple accumulators in reductions, "
  "and tile transposes. On an 8 vCPU Xeon, changed ops run 2.3x to 63x faster than upstream. "
  "Softmax is 21x faster, exp 26x, sums and max about 20x, transposes 20x, sort 7x, and add/mul about 3.5x. "
  "Test results match upstream, plus 11 new targeted tests.")
P("Hardware and build", H2)
P("Intel Xeon VM: 8 vCPU (1 thread per core), L2 16 MiB total, L3 320 MiB, 16 GB RAM, Linux, no GPU. "
  "GCC 13 (conda-forge) at -O3 with the default x86-64 target, OpenBLAS. CPU-only build (Metal and CUDA off), "
  "Release, shared library, Python 3.12 bindings. The baseline and the branch were built identically.")
P("Method", H2)
P("benchmarks/python/cpu_ops_bench.py runs 5 warmup evaluations per op, then 15 rounds of 10 mx.eval calls, "
  "and reports the median per-call time with the min..max of the per-round means. Inputs are fixed-seed and reused. "
  "The baseline was run twice. 'Before' is the second run, and the first run is shown for scale. "
  "Memory-bound single-threaded ops varied by up to ~15% between the two baseline runs. "
  "'1 thread' is the branch with MLX_CPU_NUM_THREADS=1, which isolates the single-threaded changes.")
st.append(PageBreak())
P("Results (median ms per call)", H2)
data = [["Op", "Before\n(run 1)", "Before\n(run 2)", "After", "Speedup", "After\n1 thread", "Speedup\n1 thread"]]
for n in b:
    data.append([n, f"{b1[n]['median_ms']:.3f}", f"{b[n]['median_ms']:.3f}", f"{a[n]['median_ms']:.3f}",
                 f"{b[n]['median_ms']/a[n]['median_ms']:.2f}x", f"{o[n]['median_ms']:.3f}",
                 f"{b[n]['median_ms']/o[n]['median_ms']:.2f}x"])
tbl(data, [5.2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.6*cm, 1.8*cm, 1.7*cm])
st.append(Spacer(1, 6))
P("Every changed op improves by more than the noise. gather (unchanged) and add 256x256 (kept single-threaded on "
  "purpose, below the 128 KB-per-thread grain) are about 1.0x. Single-threaded, no op got slower beyond the spread "
  "of the two baseline runs. Arrays of 64 MB fit in this machine's 320 MiB L3, so memory-bound gains are higher "
  "than on machines with small caches.", small)
st.append(PageBreak())
P("Speedup chart", H2)
st.append(Image("docs/perf/cpu_speedup.png", width=16*cm, height=16*cm*1030/930))
st.append(PageBreak())
P("Changes (one commit each)", H2)
ch = [["Commit", "Change"],
 ["1ff67f3", "Fork-join worker pool + parallel_for: generation-tagged tickets, nested/concurrent regions run serially,\nshort spin before sleeping, MLX_CPU_NUM_THREADS override."],
 ["1e6f08a", "Unary ops split across threads; inner loop kept out of line so GCC vectorizes it."],
 ["7b742b9", "Binary ops (contiguous and broadcast) split across threads."],
 ["81f05c1", "Ternary ops (where) split across threads."],
 ["71fff91", "32x32 tiled transpose-like copies; copies and fills split across threads."],
 ["c312552", "Simd math vectorizable with GCC on x86: multiply-add instead of libm std::fma without FMA,\nexact branch-free float floor, -fno-trapping-math on GCC (clang's default; results unchanged)."],
 ["47667fe", "8 independent accumulators in contiguous reductions without a SIMD type (also more accurate)."],
 ["4994380", "Reductions split across threads (all-reduce, rows, strided blocks/columns)."],
 ["eb2a1db", "Softmax rows split across threads."],
 ["25766a1", "Sort/argsort rows in parallel."],
 ["fe0fa8e", "Targeted tests vs NumPy; benchmark script."]]
tbl(ch, [1.8*cm, 14.5*cm], fs=7.5, right=False)
P("Correctness", H2)
tbl([["Suite", "Upstream 96fbf67", "perf/cpu"],
     ["C++ tests (each case separately)", "225 / 243 pass", "225 / 243 pass (same 18 failures)"],
     ["Python tests", "775 run: 8 fail, 1 error, 60 skip", "786 run: same 8 fail + 1 error, 60 skip"],
     ["New test_cpu_parallel.py", "10 / 11 (fails sum accuracy)", "11 / 11"]],
    [5.5*cm, 5.2*cm, 5.6*cm], fs=8, right=False)
st.append(Spacer(1, 4))
P("The shared failures are the mx.compile tests (all 18 in C++, 8 of 9 in Python) plus test_vmap.test_broadcast_axes_vmap. "
  "They fail identically on upstream in this environment. "
  "exp's max relative error over [-87, 87] is 3.85e-6 on both builds. "
  "The 16M-element float32 sum error vs float64 drops from 1.1e-4 to 8.5e-7. "
  "Outputs with 1 and 4 threads are bit-identical for elementwise ops, copies and sorts. "
  "Reassociated float reductions can differ from upstream in the last bits.")
P("Limits", H2)
for t in [
 "CPU only, Linux x86 only. No Metal, CUDA or Apple Silicon numbers. The Apple Silicon behaviour of the thread pool is untested.",
 "The fma/floor/-fno-trapping-math change only affects GCC x86 builds without FMA/SSE4.1 enabled at compile time.",
 "mx.compile fused CPU kernels don't work in this environment on upstream either, so compiled graphs weren't benchmarked.",
 "One machine, an 8 vCPU VM with a very large L3. The parallel grain is a fixed constant, not tuned per op.",
 "Intermediate commits weren't built one at a time; the full branch was built and tested.",
 "Not touched: BLAS/matmul, conv, gather/scatter, scans, quantized ops."]:
    P("&bull; " + t)
P("Reproduce", H2)
P("Build with cmake -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_PYTHON_BINDINGS=ON. "
  "Then run python benchmarks/python/cpu_ops_bench.py --json after.json, do the same on upstream to get before.json, "
  "and run python benchmarks/python/cpu_ops_report.py before.json after.json --chart speedup.png. "
  "Full commands are in PERF.md.", small)
SimpleDocTemplate("docs/mlx-cpu-perf.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                  topMargin=1.6*cm, bottomMargin=1.6*cm, title="MLX CPU backend perf/cpu",
                  author="shubhxho").build(st)
