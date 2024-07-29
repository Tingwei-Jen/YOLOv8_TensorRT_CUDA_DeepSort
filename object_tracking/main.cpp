#include <iostream>
#include <filesystem>
#include "engine.h"
#include "detector.h"
#include "extractor.h"
#include "tracker.h"
#include "tic_toc.h"

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
    config.probabilityThreshold = 0.65;
    config.nmsThreshold = 0.5;
    Detector detector(detectorEnginePath, config);

    // declare extractor
    Extractor extractor(extractorEnginePath, imgWidth, imgHeight);


    // declare tracker
    TrackerOptions trackerOptions;
    trackerOptions.budget = 10;
    trackerOptions.cos_method = true;
    trackerOptions.matching_threshold = 0.15;
    trackerOptions.max_iou_distance = 0.7;
    trackerOptions.max_age = 20;
    trackerOptions.n_init = 3;
    Tracker tracker(trackerOptions);

    std::vector<TrackerResult> trackingResults;

    for (int i = 0; i < nImages; i++) {
        std::string imagePath = imagePaths[i];
        std::cout << "Processing image: " << imagePath << std::endl;

        // Read the input image
        auto cpuImg = cv::imread(imagePath);
        if (cpuImg.empty()) {
            std::cerr << "Unable to read image at path: " << imagePath << std::endl;
            continue;
        }

        // detect
        std::vector<DetectionInfer> detections = detector.detect(cpuImg);

        // Extract features
        extractor.extract(cpuImg, detections);

        // Tracker update
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

        tracker.predict();
        tracker.update(detections_deepsort);
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

    return 0;
}