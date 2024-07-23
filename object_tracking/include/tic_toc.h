#ifndef TICTOC_H
#define TICTOC_H

#include <chrono>

/**
 * @brief A simple timer class for measuring durations.
 */
class TicToc {
public:
    /**
     * @brief Constructs a TicToc object and starts the timer.
     */
    TicToc() {
        start = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Calculates the duration in microseconds since the timer was started.
     * @return The duration in microseconds.
     */
    double duration() {
        end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start; ///< The start time of the timer.
    std::chrono::high_resolution_clock::time_point end;   ///< The end time of the timer.
};

#endif // TICTOC_H