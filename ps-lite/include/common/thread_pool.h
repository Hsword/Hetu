#pragma once

#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

class ThreadPool {
public:
    ThreadPool(size_t thread_num);
    ~ThreadPool();
    static ThreadPool *Get();

    template <class F, class... Args>
    auto Enqueue(F &&f, Args &&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (terminate_)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks_.emplace([task]() { (*task)(); });
        }
        cond_.notify_one();
        return res;
    }

    void Wait(int task_num);

    size_t ThreadNum() {
        return thread_num_;
    }

private:
    bool terminate_;
    size_t thread_num_;
    std::atomic_int complete_task_num_;
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cond_;
};
