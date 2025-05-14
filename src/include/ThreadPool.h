#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <future>
#include <atomic>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::vector<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<int> active_tasks;

public:
    ThreadPool(size_t threads) : stop(false), active_tasks(0) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(this->tasks.back());
                        this->tasks.pop_back();
                    }
                    
                    this->active_tasks++;
                    task();
                    this->active_tasks--;
                }
            });
        }
    }
    
    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace_back([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return res;
    }
    
    // 等待所有任务完成
    void waitAll() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if (tasks.empty() && active_tasks == 0) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

#endif /* THREAD_POOL_H_ */