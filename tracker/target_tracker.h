#include <iostream>
#include <vector>
#include <array>
#include <memory>

#include "../det_structs.h"
#include "kalman_filter.h"

constexpr size_t kDimx = 8;
constexpr size_t kDimz = 4;
constexpr float kStateCovMatBase = 1.0 / 20;
constexpr float kLinearCovMatBase = 1.0 / 160;
constexpr float kMaxAssignmentCost = 1.0;

enum kTrackState {
    New,
    Active,
    Lost,
    Removed
};

class Track {
public:
    Track() {};

public:
    virtual Bbox2D State2Bbox() const = 0;
    virtual void Initialize(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& track_id, const int32_t& frame_id) = 0;
    virtual void Predict() = 0;
    virtual void Update(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& frame_id) = 0;

public:
    int32_t GetTrackId() const {
        return track_id_;
    };
    int32_t GetLastFrameId() const {
        return frame_id_;
    };
    int32_t GetTrackletLength() const {
        return tracklet_len_;
    };
    kTrackState GetTrackState() const {
        return track_state_;
    }
    cv::Mat GetTrackMask() const {
        return mask_;
    }
    void MarkTrackState(const kTrackState& state) {
        track_state_ = state;
    }

protected:
    int32_t track_id_;
    int32_t frame_id_;
    int32_t tracklet_len_;
    kTrackState track_state_;
    KalmanFilter<kDimx, kDimz> kf_;

protected:
    int32_t cls_id_;
    std::string cls_name_;
    float det_score_;
    cv::Mat mask_;
};

class XyahTrack : public Track{
public:
    Bbox2D State2Bbox() const override;
    void Initialize(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& track_id, const int32_t& frame_id) override;
    void Predict() override;
    void Update(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& frame_id);
};

class XywhTrack : public Track{
public:
    Bbox2D State2Bbox() const override;
    void Initialize(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& track_id, const int32_t& frame_id) override;
    void Predict() override;
    void Update(const Bbox2D& bbox, const cv::Mat& mask, const int32_t& frame_id);
};

class Tracker {
public:
    Tracker(const float& split_thresh_score, const float& add_thresh_score, const float& cost_thresh_iou, const int32_t& max_lost_frame_num):
        split_thresh_score_(split_thresh_score),
        add_thresh_score_(add_thresh_score),
        cost_thresh_iou_(cost_thresh_iou),
        max_lost_frame_num_(max_lost_frame_num),
        cur_frame_id_(0),
        all_tracks_num_(0)
    {}
    Tracker(const int32_t& max_lost_frame_num):
        split_thresh_score_(0.5),   // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        add_thresh_score_(0.6),     // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        cost_thresh_iou_(0.0),      // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        max_lost_frame_num_(max_lost_frame_num),
        cur_frame_id_(0),
        all_tracks_num_(0)
    {}
    Tracker():
        split_thresh_score_(0.5),   // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        add_thresh_score_(0.6),     // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        cost_thresh_iou_(0.0),      // Reference https://github.com/ultralytics/ultralytics/tracker/cfg/bytetrack.yaml
        max_lost_frame_num_(50),    // Should be the same with input stream's fps
        cur_frame_id_(0),
        all_tracks_num_(0)
    {}

public:
    void Initialize();
    void Update(const std::vector<Bbox2D>& det_boxes, std::vector<const Bbox2D*>& unmatched_boxes);
    void Update(const std::vector<Bbox2D>& det_boxes, const std::vector<cv::Mat>& mask_mats, std::vector<const Bbox2D*>& unmatched_boxes);

public:
    const std::vector<std::shared_ptr<Track>>* GetNewTracklets() const {
        return &new_tracks_;
    }
    const std::vector<std::shared_ptr<Track>>* GetActiveTracklets() const {
        return &active_tracks_;
    }

private:
    void CalculateCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<Bbox2D>& col_boxes, std::vector<std::vector<float>>& cost_matrix);
    void LinearAssignment(const std::vector<Track>& cur_tracklets, const std::vector<Bbox2D>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res);
    void CalculateCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::vector<float>>& cost_matrix);
    void LinearAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res);

private:
    int32_t cur_frame_id_;
    int32_t all_tracks_num_;

private:
    float split_thresh_score_;      // The threshold of cls score to split high_conf and low_conf detection boxes
    float add_thresh_score_;        // The threshold of cls score determining whether to add a det box into new_tracks_ vector
    float cost_thresh_iou_;         // The threshold of iou detereming whether two boxes would ever be matched
    int32_t max_lost_frame_num_;    // The threshold of frame_num to decide the lost boxes' max alive time

private:
    std::vector<std::shared_ptr<Track>> new_tracks_;
    std::vector<std::shared_ptr<Track>> active_tracks_;
    std::vector<std::shared_ptr<Track>> lost_tracks_;
    std::vector<std::shared_ptr<Track>> removed_tracks_;
};