#include "common/thread_pool.h"

static ThreadPool *pool;
const size_t kThreadNum = 5;

ThreadPool::ThreadPool(size_t thread_num) :
    terminate_(false), thread_num_(thread_num), complete_task_num_(0) {
    for (size_t i = 0; i < thread_num; ++i) {
        threads_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->mutex_);
                    this->cond_.wait(lock, [this] {
                        return this->terminate_ || !this->tasks_.empty();
                    });

                    if (this->terminate_ && this->tasks_.empty())
                        return;

                    task = std::move(this->tasks_.front());
                    this->tasks_.pop();
                }
                task();
                complete_task_num_++;
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        terminate_ = true;
    }
    cond_.notify_all();

    for (std::thread &thread : threads_) {
        thread.join();
    }
}

void ThreadPool::Wait(int task_num) {
    while (complete_task_num_ != task_num) {
        usleep(1000);
    }
    complete_task_num_ = 0;
}

ThreadPool *ThreadPool::Get() {
    if (!pool) {
        pool = new ThreadPool(kThreadNum);
    }
    return pool;
}
