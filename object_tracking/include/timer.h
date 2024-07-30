#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <functional>
#include <string>

class Timer {
public:
    using Callback = std::function<void(const std::string&, long long)>;

    Timer(const std::string& functionName, Callback callback)
        : functionName_(functionName), callback_(callback) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        long long duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        callback_(functionName_, duration);
    }

private:
    std::string functionName_; ///< The name of the function that is being timed.
    Callback callback_; ///< The callback function that is called when the timer is stopped.
    std::chrono::high_resolution_clock::time_point start_; ///< The start time of the timer.
};

#endif // TIMER_H