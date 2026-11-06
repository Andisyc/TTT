#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include "threadpool.h"

using mujoco::python::ThreadPool;

int main() {
  ThreadPool pool(2);
  std::atomic<int> completed{0};
  pool.ResetCount();
  pool.Schedule([] { throw std::runtime_error("worker failure sentinel"); });
  pool.Schedule([&] { completed.fetch_add(1); });
  try {
    pool.WaitCount(2);
    throw std::logic_error("worker error was not propagated");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()) != "worker failure sentinel" || completed != 1) {
      throw std::logic_error("worker error returned before sibling retirement");
    }
  }
  {
    ThreadPool::TaskBatch batch(pool);
    batch.Schedule([&] { completed.fetch_add(1); });
    batch.Wait();
  }
  if (completed != 2) throw std::logic_error("pool did not recover after worker error");
  std::cout << "PASS: worker failure drained, propagated, and pool reused\n";

  try {
    ThreadPool::TaskBatch batch(pool);
    batch.Schedule([&] { completed.fetch_add(1); });
    throw std::runtime_error("submission failure sentinel");
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()) != "submission failure sentinel" || completed != 3) {
      throw std::logic_error("submission unwind did not retire submitted work");
    }
  }
  std::cout << "PASS: submission unwind retired work before releasing caller storage\n";

  std::atomic<int> retired{0};
  for (int cycle = 0; cycle < 20; ++cycle) {
    ThreadPool temporary(2);
    for (int task = 0; task < 8; ++task) {
      temporary.Schedule([&] { retired.fetch_add(1); });
    }
  }
  if (retired != 160) throw std::logic_error("pool destruction abandoned work");
  std::cout << "PASS: repeated create/use/destroy joined all scheduled work\n";
}
