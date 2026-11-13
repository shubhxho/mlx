// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/threading.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace mlx::core::cpu {

namespace {

constexpr auto kSpinTime = std::chrono::microseconds(100);

inline void cpu_relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

// Set while a thread is executing a chunk of a parallel region, so that
// nested parallel_for calls run serially instead of deadlocking.
thread_local bool in_parallel_region = false;

int default_num_threads() {
  if (const char* env = std::getenv("MLX_CPU_NUM_THREADS")) {
    int n = std::atoi(env);
    if (n > 0) {
      return n;
    }
  }
  int n = static_cast<int>(std::thread::hardware_concurrency());
  return std::max(n, 1);
}

class ThreadPool {
 public:
  explicit ThreadPool(int n_workers) {
    workers_.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_.store(true);
    }
    cv_.notify_all();
    for (auto& t : workers_) {
      t.join();
    }
  }

  // Returns false if the pool is busy with another region.
  bool run(int64_t n, int nchunks, detail::RangeFn fn, void* ctx) {
    std::unique_lock<std::mutex> region_lock(region_mtx_, std::try_to_lock);
    if (!region_lock.owns_lock()) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(mtx_);
      fn_ = fn;
      ctx_ = ctx;
      n_ = n;
      nchunks_.store(nchunks, std::memory_order_relaxed);
      uint64_t gen = generation_.load(std::memory_order_relaxed) + 1;
      pending_.store(nchunks, std::memory_order_relaxed);
      // Publishing the ticket (generation, chunk 0) releases the fields above
      // to any thread that claims a chunk of this region.
      ticket_.store(gen << 32, std::memory_order_release);
      generation_.store(gen, std::memory_order_release);
    }
    cv_.notify_all();

    // The caller participates in the work.
    in_parallel_region = true;
    run_chunks(generation_.load(std::memory_order_relaxed));
    in_parallel_region = false;

    // Wait for the chunks claimed by workers. Spin briefly since chunks are
    // roughly equal-sized and will finish soon, then block.
    for (int i = 0; i < 2000; ++i) {
      if (pending_.load(std::memory_order_acquire) == 0) {
        return true;
      }
      std::this_thread::yield();
    }
    std::unique_lock<std::mutex> lk(mtx_);
    done_cv_.wait(
        lk, [this] { return pending_.load(std::memory_order_acquire) == 0; });
    return true;
  }

 private:
  // Claim and run chunks of region `gen` until none are left. A ticket packs
  // the region generation (high 32 bits) with the next chunk index, so that a
  // worker that woke up late can never claim a chunk of a newer region.
  void run_chunks(uint64_t gen) {
    gen &= 0xffffffffu;
    uint64_t t = ticket_.load(std::memory_order_acquire);
    while (true) {
      if ((t >> 32) != gen) {
        return;
      }
      // May be stale if `t` is stale, in which case the CAS below fails.
      int nchunks = nchunks_.load(std::memory_order_relaxed);
      int c = static_cast<int>(t & 0xffffffffu);
      if (c >= nchunks) {
        return;
      }
      if (!ticket_.compare_exchange_weak(
              t, t + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      int64_t begin = n_ * c / nchunks;
      int64_t end = n_ * (c + 1) / nchunks;
      fn_(ctx_, begin, end);
      if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // Last chunk: wake a blocked caller (if any).
        std::lock_guard<std::mutex> lk(mtx_);
        done_cv_.notify_all();
      }
      t = ticket_.load(std::memory_order_acquire);
    }
  }

  void worker_loop() {
    in_parallel_region = true;
    uint64_t seen = 0;
    while (true) {
      // Kernels are usually issued back to back, so spin for a short while
      // before going to sleep to avoid paying the wake-up latency for every
      // parallel region.
      auto spin_until = std::chrono::steady_clock::now() + kSpinTime;
      int iters = 0;
      while (generation_.load(std::memory_order_acquire) == seen &&
             !stop_.load(std::memory_order_relaxed)) {
        if (++iters % 64 == 0 &&
            std::chrono::steady_clock::now() > spin_until) {
          break;
        }
        cpu_relax();
      }
      if (uint64_t g = generation_.load(std::memory_order_acquire); g != seen) {
        seen = g;
        run_chunks(seen);
        continue;
      }
      {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&] {
          return stop_.load(std::memory_order_relaxed) ||
              generation_.load(std::memory_order_relaxed) != seen;
        });
        if (stop_.load(std::memory_order_relaxed)) {
          return;
        }
        seen = generation_.load(std::memory_order_relaxed);
      }
      run_chunks(seen);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex region_mtx_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> generation_{0};

  // Current region (written under mtx_ before generation_ is bumped).
  detail::RangeFn fn_{nullptr};
  void* ctx_{nullptr};
  int64_t n_{0};
  std::atomic<int> nchunks_{0};
  std::atomic<uint64_t> ticket_{0};
  std::atomic<int> pending_{0};
};

ThreadPool& pool() {
  // Intentionally leaked: worker threads must not be joined during static
  // destruction while stream threads may still be running kernels.
  static ThreadPool* p = new ThreadPool(max_threads() - 1);
  return *p;
}

} // namespace

int max_threads() {
  static int n = default_num_threads();
  return n;
}

namespace detail {

void parallel_for_impl(int64_t n, int nchunks, RangeFn fn, void* ctx) {
  if (in_parallel_region || nchunks <= 1 || !pool().run(n, nchunks, fn, ctx)) {
    fn(ctx, 0, n);
  }
}

} // namespace detail

} // namespace mlx::core::cpu
