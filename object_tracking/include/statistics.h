#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <map>
#include <string>

class Statistics {
public:
    void addDuration(const std::string& functionName, long long duration) {
        durations_[functionName].push_back(duration);
    }

    long long getAverageDuration(const std::string& functionName) {
        long long sum = 0;
        for (long long duration : durations_[functionName]) {
            sum += duration;
        }
        return durations_[functionName].size() ? sum / durations_[functionName].size() : 0;
    }

    std::map<std::string, std::vector<long long>> getDurations() {
        return durations_;
    }

private:
    std::map<std::string, std::vector<long long>> durations_;
};

#endif // STATISTICS_H