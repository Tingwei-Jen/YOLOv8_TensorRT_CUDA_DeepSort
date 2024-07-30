#include <iostream>
#include <filesystem>
#include "engine.h"
#include "detector.h"
#include "extractor.h"
#include "tracker.h"
#include "statistics.h"

std::vector<std::string> getImagePaths(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    
    // Check if the provided path is a directory
    if (!std::filesystem::is_directory(folderPath)) {
        std::cerr << "Provided path is not a directory." << std::endl;
        return imagePaths;
    }

    // Iterate through the directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        // Check if it's a file and has an image extension
        if (std::filesystem::is_regular_file(entry.status())) {
            std::string path = entry.path().string();
            std::string extension = entry.path().extension().string();
            
            // Check for common image file extensions
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" || extension == ".gif") {
                imagePaths.push_back(path);
            }
        }
    }

    // Sort the image paths
    std::sort(imagePaths.begin(), imagePaths.end());
    
    return imagePaths;
}

struct TrackerResult {
    std::string imgPath;
    int classId;
    int trackId;
    float confidence;
    Eigen::Vector4f tlwh;
};

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <detector_engine_path> <extractor_engine_path> <images_folder_path> <tracking_result_output_folder_path>" << std::endl;
        return 1;
    }

    // Get the paths
    std::string detectorEnginePath = argv[1];
    std::string extractorEnginePath = argv[2];
    std::string imagesFolderPath = argv[3];
    std::string trackerResultOutputFolderPath = argv[4];

    // get images paths
    std::vector<std::string> imagePaths = getImagePaths(imagesFolderPath);
    int nImages = imagePaths.size();
    if (nImages == 0) {
        std::cerr << "No images found in the provided folder." << std::endl;
        return 1;
    }
    
    // Read the first image to get the image dimensions
    auto testImg = cv::imread(imagePaths[0]);
    
    if (testImg.empty()) {
        std::cerr << "Unable to read image at path: " << imagePaths[0] << std::endl;
        return 1;
    }

    int imgWidth = testImg.cols;
    int imgHeight = testImg.rows;

    // declare detector
    DetectorConfig config;
    config.imgWidth = imgWidth;
    config.imgHeight = imgHeight;
    config.probabilityThreshold = 0.5;
    config.nmsThreshold = 0.5;
    Statistics statistics;
    Detector detector(detectorEnginePath, config, statistics);

    // declare extractor
    Extractor extractor(extractorEnginePath, imgWidth, imgHeight, statistics);

    // declare tracker
    TrackerOptions trackerOptions;
    trackerOptions.budget = 50;
    trackerOptions.cos_method = true;
    trackerOptions.matching_threshold = 0.15;
    trackerOptions.max_iou_distance = 0.7;
    trackerOptions.max_age = 30;
    trackerOptions.n_init = 3;
    Tracker tracker(trackerOptions);

    std::vector<TrackerResult> trackingResults;

    for (int i = 0; i < nImages; i++) {
        Timer timer("Pipeline", [&statistics](const std::string& functionName, long long duration){
            statistics.addDuration(functionName, duration);
        });

        std::string imagePath = imagePaths[i];
        std::cout << "Processing image: " << imagePath << std::endl;

        // Read the input image
        cv::Mat cpuImg;
        {
            Timer timer("cv::imread", [&statistics](const std::string& functionName, long long duration){
                statistics.addDuration(functionName, duration);
            });

            cpuImg = cv::imread(imagePath);
            if (cpuImg.empty()) {
                std::cerr << "Unable to read image at path: " << imagePath << std::endl;
                continue;
            }
        }

        // detect
        std::vector<DetectionInfer> detections = detector.detect(cpuImg);

        // Extract features
        extractor.extract(cpuImg, detections);

        // detection to deepsort format
        std::vector<DetectionSort> detections_deepsort;

        for (auto det: detections) {
            if (det.getCls() != 0)
                continue; 
            Eigen::Vector4f tlwh = det.get_tlwh();
            float confidence = det.getConfidence();
            Eigen::VectorXf feature = det.getFeature();
            int cls = det.getCls();
            DetectionSort det_sort(tlwh, confidence, cls, feature);
            detections_deepsort.push_back(det_sort);
        }

        // Update tracker
        {
            Timer timer("Tracker::update", [&statistics](const std::string& functionName, long long duration){
                statistics.addDuration(functionName, duration);
            });
            tracker.predict();
            tracker.update(detections_deepsort);
        }
        
        // output tracking results
        std::vector<Track> tracks = tracker.getTracks();
        for (auto track: tracks) {
            if (!track.is_confirmed() || track.time_since_update() > 1) {
                continue;
            }
            TrackerResult res;
            res.imgPath = imagePath;
            res.classId = track.cls();
            res.trackId = track.track_id();
            res.confidence = track.confidence();
            res.tlwh = track.to_tlwh();
            trackingResults.push_back(res);
        }
    }

    // Print the duration statistics
    auto durations = statistics.getDurations();
    for (const auto& entry : durations) {
        const auto& functionName = entry.first;
        const auto& times = entry.second;
        long long total = 0;
        for (auto time : times) {
            total += time;
        }
        long long average = times.empty() ? 0 : total / times.size();
        std::cout << "Function " << functionName << " called " << times.size() << " times. "
                  << "Average duration: " << average << " microseconds.\n";
    }

    // Write the tracking results to a file
    std::string trackerResultOutputFilePath = trackerResultOutputFolderPath + "tracker_result.txt";
    std::ofstream outfile(trackerResultOutputFilePath);
    outfile << "img_path, class_id, track_id, confidence, tlx, tly, w, h" << std::endl;
    for (auto res: trackingResults) {
        outfile << res.imgPath << "," << res.classId << "," << res.trackId << "," << res.confidence << ",";
        outfile << res.tlwh(0) << ","; // top left x
        outfile << res.tlwh(1) << ","; // top left y
        outfile << res.tlwh(2) << ","; // width
        outfile << res.tlwh(3) << std::endl; // height
    }
    outfile.close();

    // Write the duration results to a file
    std::string durationResultOutputFilePath = trackerResultOutputFolderPath + "duration_result.txt";
    std::cout << "Writing duration results to: " << durationResultOutputFilePath << std::endl;
    std::ofstream durationOutfile(durationResultOutputFilePath);
    durationOutfile << "img_path, Pipeline, cv::imread, Detector::preprocessing, Detector::inference, Detector::postprocessing, Detector::outputDets, Extractor::extract, Extractor::inference, Tracker::update" << std::endl;
    int nData = durations["Pipeline"].size();

    for (int i = 0; i < nData; i++) {
        durationOutfile << imagePaths[i] << ",";
        durationOutfile << durations["Pipeline"][i] << ",";
        durationOutfile << durations["cv::imread"][i] << ",";
        durationOutfile << durations["Detector::preprocessing"][i] << ",";
        durationOutfile << durations["Detector::inference"][i] << ",";
        durationOutfile << durations["Detector::postprocessing"][i] << ",";
        durationOutfile << durations["Detector::outputDets"][i] << ",";
        durationOutfile << durations["Extractor::extract"][i] << ",";
        durationOutfile << durations["Extractor::inference"][i] << ",";
        durationOutfile << durations["Tracker::update"][i] << std::endl;
    }

    durationOutfile.close();
    return 0;
}