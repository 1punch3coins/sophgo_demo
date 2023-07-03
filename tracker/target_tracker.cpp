#include <algorithm>
#include "target_tracker.h"
#include "hungarian_algorithm.h"

void XyahTrack::Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) {
    // Use xyah as the first four state variables, the others are their correspondant velocity, initialized as zero 
    EigenVector<kDimx> X;
    X << bbox.x, bbox.y, static_cast<float>(bbox.w) / bbox.h, bbox.h, 0, 0, 0, 0;
    // Initalize X's convariance matrix
    EigenMatrix<kDimx, kDimx> P;
    const float v1 =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
    const float v2 =  kLinearCovMatBase * kLinearCovMatBase * X[3] * X[3];
    P << 4*v1, 0, 0, 0, 0, 0, 0, 0,
         0, 4*v1, 0, 0, 0, 0, 0, 0,
         0, 0, 1e-4, 0, 0, 0, 0, 0,
         0, 0, 0, 4*v1, 0, 0, 0, 0,
         0, 0, 0, 0, 100*v2, 0, 0, 0,
         0, 0, 0, 0, 0, 100*v2, 0, 0,
         0, 0, 0, 0, 0, 0, 1e-10, 0,
         0, 0, 0, 0, 0, 0, 0, 100*v2;
    // Initialize uniform linear transition matrix (X(t-1)--->X(t))
    EigenMatrix<kDimx, kDimx> F;
    F << 1, 0, 0, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 0, 1, 0, 0, 0, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 1;
    // Initialize project transistion matrix (X(t)--->X(t))
    EigenMatrix<kDimz, kDimx> H;
    H << 1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0;
    kf_ = KalmanFilter<kDimx, kDimz>(X, P, F, H);
    track_id_ = track_id;
    frame_id_ = frame_id;
    cls_id_ = bbox.cls_id;
    tracklet_len_ = 0;
    det_score_ = bbox.cls_confidence;
    track_state_ = kTrackState::New;
}

void XyahTrack::Predict() {
     EigenMatrix<kDimx, kDimx> Q;  // Enviroment uncertentiy
     const EigenVector<kDimx> X = kf_.GetStateMean();
     const float v1 =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
     const float v2 =  kLinearCovMatBase * kLinearCovMatBase * X[3] * X[3];
     Q << v1, 0, 0, 0, 0, 0, 0, 0,
          0, v1, 0, 0, 0, 0, 0, 0,
          0, 0,1e-4,0, 0, 0, 0, 0,
          0, 0, 0, v1, 0, 0, 0, 0,
          0, 0, 0, 0, v2, 0, 0, 0,
          0, 0, 0, 0, 0, v2, 0, 0,
          0, 0, 0, 0, 0, 0,1e-10,0,
          0, 0, 0, 0, 0, 0, 0, v2;
     kf_.Predict(Q);
}

void XyahTrack::Update(const Bbox2D& bbox, const int32_t& frame_id) {
     EigenVector<kDimz> Z;    // New meassurement
     EigenMatrix<kDimz, kDimz> R;  // New measurment's uncertentiy
     const EigenVector<kDimx> X = kf_.GetStateMean();
     Z << bbox.x, bbox.y, static_cast<float>(bbox.w) / bbox.h, bbox.h;
     const float v =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
     R << v, 0, 0, 0,
          0, v, 0, 0,
          0, 0, 1e-2, 0,
          0, 0, 0, v;
     kf_.Update(Z, R);
     tracklet_len_++;
     cls_id_ = bbox.cls_id;
     det_score_ = bbox.cls_confidence;
     cls_name_ = bbox.cls_name;
     frame_id_ = frame_id;
}

Bbox2D XyahTrack::State2Bbox() const {
     const EigenVector<kDimx> X = kf_.GetStateMean();
     Bbox2D bbox;
     bbox.x = static_cast<int32_t>(X[0]);
     bbox.y = static_cast<int32_t>(X[1]);
     bbox.w = static_cast<int32_t>(X[2] * X[3]);
     bbox.h = static_cast<int32_t>(X[3]);
     bbox.cls_id = cls_id_;
     bbox.cls_confidence = det_score_;
     bbox.cls_name = cls_name_;
     return bbox;
}

void XywhTrack::Initialize(const Bbox2D& bbox, const int32_t& track_id, const int32_t& frame_id) {
    // Use xywh as the first four state variables, the others are their correspondant velocity, initialized as zero 
    EigenVector<kDimx> X;
    X << bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, 0, 0;
    // Initalize X's convariance matrix
    EigenMatrix<kDimx, kDimx> P;
    const float v1 =  kStateCovMatBase * kStateCovMatBase * X[2] * X[2];
    const float v2 =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
    const float v3 =  kLinearCovMatBase * kLinearCovMatBase * X[2] * X[2];
    const float v4 =  kLinearCovMatBase * kLinearCovMatBase * X[3] * X[3];
    P << 4*v1, 0, 0, 0, 0, 0, 0, 0,
         0, 4*v2, 0, 0, 0, 0, 0, 0,
         0, 0, 4*v1, 0, 0, 0, 0, 0,
         0, 0, 0, 4*v2, 0, 0, 0, 0,
         0, 0, 0, 0, 100*v1, 0, 0, 0,
         0, 0, 0, 0, 0, 100*v2, 0, 0,
         0, 0, 0, 0, 0, 0, 100*v1, 0,
         0, 0, 0, 0, 0, 0, 0, 100*v2;
    // Initialize uniform linear transition matrix (X(t-1)--->X(t))
    EigenMatrix<kDimx, kDimx> F;
    F << 1, 0, 0, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 0, 1, 0, 0, 0, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 1,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 0, 0, 1;
    // Initialize project transistion matrix (X(t)--->X(t))
    EigenMatrix<kDimz, kDimx> H;
    H << 1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0;
    kf_ = KalmanFilter<kDimx, kDimz>(X, P, F, H);
    track_id_ = track_id;
    frame_id_ = frame_id;
    cls_id_ = bbox.cls_id;
    tracklet_len_ = 0;
    det_score_ = bbox.cls_confidence;
    track_state_ = kTrackState::New;
}

void XywhTrack::Predict() {
     EigenMatrix<kDimx, kDimx> Q;  // Enviroment uncertentiy
     const EigenVector<kDimx> X = kf_.GetStateMean();
     const float v1 =  kStateCovMatBase * kStateCovMatBase * X[2] * X[2];
     const float v2 =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
     const float v3 =  kLinearCovMatBase * kLinearCovMatBase * X[2] * X[2];
     const float v4 =  kLinearCovMatBase * kLinearCovMatBase * X[3] * X[3];
     Q << v1, 0, 0, 0, 0, 0, 0, 0,
          0, v2, 0, 0, 0, 0, 0, 0,
          0, 0, v1, 0, 0, 0, 0, 0,
          0, 0, 0, v2, 0, 0, 0, 0,
          0, 0, 0, 0, v3, 0, 0, 0,
          0, 0, 0, 0, 0, v3, 0, 0,
          0, 0, 0, 0, 0, 0, v4, 0,
          0, 0, 0, 0, 0, 0, 0, v4;
     kf_.Predict(Q);
}

void XywhTrack::Update(const Bbox2D& bbox, const int32_t& frame_id) {
     EigenVector<kDimz> Z;    // New meassurement
     EigenMatrix<kDimz, kDimz> R;  // New measurment's uncertentiy
     const EigenVector<kDimx> X = kf_.GetStateMean();
     Z << bbox.x, bbox.y, bbox.w, bbox.h;
     const float v1 =  kStateCovMatBase * kStateCovMatBase * X[2] * X[2];
     const float v2 =  kStateCovMatBase * kStateCovMatBase * X[3] * X[3];
     R << v1, 0, 0, 0,
          0, v2, 0, 0,
          0, 0, v1, 0,
          0, 0, 0, v2;
     kf_.Update(Z, R);
     tracklet_len_++;
     cls_id_ = bbox.cls_id;
     det_score_ = bbox.cls_confidence;
     cls_name_ = bbox.cls_name;
     frame_id_ = frame_id;
}

Bbox2D XywhTrack::State2Bbox() const {
     const EigenVector<kDimx> X = kf_.GetStateMean();
     Bbox2D bbox;
     bbox.x = static_cast<int32_t>(X[0]);
     bbox.y = static_cast<int32_t>(X[1]);
     bbox.w = static_cast<int32_t>(X[2]);
     bbox.h = static_cast<int32_t>(X[3]);
     bbox.cls_id = cls_id_;
     bbox.cls_confidence = det_score_;
     bbox.cls_name = cls_name_;
     return bbox;
}

void Tracker::CalculateCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<Bbox2D>& col_boxes, std::vector<std::vector<float>>& cost_matrix) {
     int32_t cost_matrix_size = (std::max)(row_boxes.size(), col_boxes.size());
     cost_matrix.resize(cost_matrix_size, std::vector<float>(cost_matrix_size, kMaxAssignmentCost));
     for (int32_t i = 0; i < row_boxes.size(); i++) {
          for (int32_t j = 0; j < col_boxes.size(); j++) {
               int32_t inter_left   = std::max(row_boxes[i].x, col_boxes[j].x);
               int32_t inter_right  = std::min(row_boxes[i].x + row_boxes[i].w, col_boxes[j].x + col_boxes[j].w);
               int32_t inter_top    = std::max(row_boxes[i].y, col_boxes[j].y);
               int32_t inter_bottom = std::min(row_boxes[i].y + row_boxes[i].h, col_boxes[j].y + col_boxes[j].h);
               if (inter_left > inter_right || inter_top > inter_bottom) { 
                    continue;
               }
               int32_t area_inter = (inter_right - inter_left) * (inter_bottom - inter_top);
               int32_t area_i = row_boxes[i].h * row_boxes[i].w;
               int32_t area_j = col_boxes[j].h * col_boxes[j].w;
               float iou = static_cast<float>(area_inter) / (area_i + area_j - area_inter);
               if (iou < cost_thresh_iou_) {
                    continue;
               }
               cost_matrix[i][j] = kMaxAssignmentCost - iou;
          }
     }
}

void Tracker::LinearAssignment(const std::vector<Track>& cur_tracklets, const std::vector<Bbox2D>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res) {
     if (cur_tracklets.size() == 0 || det_boxes.size() == 0) {
          tracklets_assignment_res.resize(cur_tracklets.size(), -1);
          dets_assignment_res.resize(det_boxes.size(), -1);
          return;
     }
     std::vector<Bbox2D> tracklet_boxes;
     std::vector<std::vector<float>> cost_matrix;
     tracklet_boxes.reserve(cur_tracklets.size());
     for (const auto& tracklet : cur_tracklets) {
          tracklet_boxes.push_back(tracklet.State2Bbox());
     }
     CalculateCostMatrix(tracklet_boxes, det_boxes, cost_matrix);

     tracklets_assignment_res.resize(cost_matrix.size(), -1);
     dets_assignment_res.resize(cost_matrix.size(), -1);
     HungarianAlgorithm<float> solver(cost_matrix);
     solver.Solve(tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < cost_matrix.size(); i++) {
          if (tracklets_assignment_res[i] >= 0) {
               if (tracklets_assignment_res[i] >= det_boxes.size() || cost_matrix[i][tracklets_assignment_res[i]] == kMaxAssignmentCost) {
                    tracklets_assignment_res[i] = -1;
               }
          }
          if (dets_assignment_res[i] >= 0) {
               if (dets_assignment_res[i] >= cur_tracklets.size() || cost_matrix[dets_assignment_res[i]][i] == kMaxAssignmentCost) {
                    dets_assignment_res[i] = -1;
               }
          }
     }
}

void Tracker::CalculateCostMatrix(const std::vector<Bbox2D>& row_boxes, const std::vector<const Bbox2D*>& col_boxes, std::vector<std::vector<float>>& cost_matrix) {
     int32_t cost_matrix_size = (std::max)(row_boxes.size(), col_boxes.size());
     cost_matrix.resize(cost_matrix_size, std::vector<float>(cost_matrix_size, kMaxAssignmentCost));
     for (int32_t i = 0; i < row_boxes.size(); i++) {
          for (int32_t j = 0; j < col_boxes.size(); j++) {
               int32_t inter_left   = std::max(row_boxes[i].x, col_boxes[j]->x);
               int32_t inter_right  = std::min(row_boxes[i].x + row_boxes[i].w, col_boxes[j]->x + col_boxes[j]->w);
               int32_t inter_top    = std::max(row_boxes[i].y, col_boxes[j]->y);
               int32_t inter_bottom = std::min(row_boxes[i].y + row_boxes[i].h, col_boxes[j]->y + col_boxes[j]->h);
               if (inter_left > inter_right || inter_top > inter_bottom) {
                    continue;
               }
               int32_t area_inter = (inter_right - inter_left) * (inter_bottom - inter_top);
               int32_t area_i = row_boxes[i].h * row_boxes[i].w;
               int32_t area_j = col_boxes[j]->h * col_boxes[j]->w;
               float iou = static_cast<float>(area_inter) / (area_i + area_j - area_inter);
               if (iou < cost_thresh_iou_) {
                    continue;
               }
               cost_matrix[i][j] = kMaxAssignmentCost - iou;
          }
     }
}

void Tracker::LinearAssignment(const std::vector<std::shared_ptr<Track>>& cur_tracklets, const std::vector<const Bbox2D*>& det_boxes, std::vector<int32_t>& tracklets_assignment_res, std::vector<int32_t>& dets_assignment_res) {
     if (cur_tracklets.size() == 0 || det_boxes.size() == 0) {
          tracklets_assignment_res.resize(cur_tracklets.size(), -1);
          dets_assignment_res.resize(det_boxes.size(), -1);
          return;
     }
     std::vector<Bbox2D> tracklet_boxes;
     std::vector<std::vector<float>> cost_matrix;
     tracklet_boxes.reserve(cur_tracklets.size());
     for (const auto& tracklet : cur_tracklets) {
          tracklet_boxes.push_back(tracklet->State2Bbox());
     }
     CalculateCostMatrix(tracklet_boxes, det_boxes, cost_matrix);

     tracklets_assignment_res.resize(cost_matrix.size(), -1);
     dets_assignment_res.resize(cost_matrix.size(), -1);
     HungarianAlgorithm<float> solver(cost_matrix);
     solver.Solve(tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < cost_matrix.size(); i++) {
          if (tracklets_assignment_res[i] >= 0) {
               if (tracklets_assignment_res[i] >= det_boxes.size() || cost_matrix[i][tracklets_assignment_res[i]] == kMaxAssignmentCost) {
                    tracklets_assignment_res[i] = -1;
               }
          }
          if (dets_assignment_res[i] >= 0) {
               if (dets_assignment_res[i] >= cur_tracklets.size() || cost_matrix[dets_assignment_res[i]][i] == kMaxAssignmentCost) {
                    dets_assignment_res[i] = -1;
               }
          }
     }
}

void Tracker::Update(const std::vector<Bbox2D>& det_boxes, std::vector<const Bbox2D*>& unmatched_boxes) {
     // 1. Predict the active and lost tracklets' next positions
     std::vector<std::shared_ptr<Track>> merged_tracklets(active_tracks_);
     merged_tracklets.insert(merged_tracklets.begin(), lost_tracks_.begin(), lost_tracks_.end());
     for (const auto& tracklet : merged_tracklets) {
          tracklet->Predict();
     }

     // 2. Split the current frame's meassurments into two sets
     std::vector<const Bbox2D*> high_conf_dets;
     std::vector<const Bbox2D*> low_conf_dets;
     for (const auto& det_box : det_boxes) {
          if (det_box.cls_confidence >= split_thresh_score_) {
               high_conf_dets.push_back(&det_box);
          } else {
               low_conf_dets.push_back(&det_box);
          }
     }

     // 3.1 First-pass assignment, mathcing between merged tracklets with high-conf dets
     std::vector<int32_t> tracklets_assignment_res;
     std::vector<int32_t> dets_assignment_res;
     std::vector<std::shared_ptr<Track>> unmatched_tracklets;
     std::vector<const Bbox2D*> unmatched_high_conf_dets;
     LinearAssignment(merged_tracklets, high_conf_dets, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < merged_tracklets.size(); i++) {
          Track* tracklet = merged_tracklets[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = high_conf_dets[tracklets_assignment_res[i]];
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    tracklet->Update(*bbox, cur_frame_id_);
               } else {
                    tracklet->Update(*bbox, cur_frame_id_);
                    // remove from lost vec & push_back into active vec
                    tracklet->MarkTrackState(kTrackState::Active);
                    active_tracks_.push_back(merged_tracklets[i]);
                    // lost_tracks_.erase(lost_tracks_.begin() + (i - active_tracks_.size())); // It will cause a bug
               }
          } else {
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    unmatched_tracklets.push_back(merged_tracklets[i]);
               }
          }
     }
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Lost) {
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (int32_t i = 0; i < high_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_high_conf_dets.push_back(high_conf_dets[i]);
          }
     }

     // 3.2 Second-pass assignment, matching between the first-pass unmatched-active tracklets with low-conf dets
     std::vector<const Bbox2D*> unmatched_low_conf_dets;
     tracklets_assignment_res.clear();
     dets_assignment_res.clear();
     LinearAssignment(unmatched_tracklets, low_conf_dets, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = 0; i < unmatched_tracklets.size(); i++) {
          Track* tracklet = unmatched_tracklets[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = low_conf_dets[tracklets_assignment_res[i]];
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    tracklet->Update(*bbox, cur_frame_id_);
               } else {
                    tracklet->Update(*bbox, cur_frame_id_);
                    // remove from lost vec & push_back into active vec
                    tracklet->MarkTrackState(kTrackState::Active);
                    active_tracks_.push_back(unmatched_tracklets[i]);
               }
          } else {
               if (tracklet->GetTrackState() == kTrackState::Active) {
                    // remove from active vec & push_back into lost vec
                    tracklet->MarkTrackState(kTrackState::Lost);
                    lost_tracks_.push_back(unmatched_tracklets[i]);
               }
          }
     }
     for (auto it = active_tracks_.begin(); it != active_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Active) {
               it = active_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (it->get()->GetTrackState() != kTrackState::Lost) {
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }
     for (int32_t i = 0; i < low_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_low_conf_dets.push_back(low_conf_dets[i]);
          }
     }
     merged_tracklets.clear();
     unmatched_tracklets.clear();
     
     // 3.3 Thrid-pass assignment, matching between last-frame's new tracklets and first-pass unmatched high-conf dets
     tracklets_assignment_res.clear();
     dets_assignment_res.clear();
     std::vector<const Bbox2D*> unmatched_twice_high_conf_dets;
     LinearAssignment(new_tracks_, unmatched_high_conf_dets, tracklets_assignment_res, dets_assignment_res);
     for (int32_t i = new_tracks_.size() - 1; i > -1; i--) {
          Track* tracklet = new_tracks_[i].get();
          if (tracklets_assignment_res[i] >= 0) {
               const Bbox2D* bbox = unmatched_high_conf_dets[tracklets_assignment_res[i]];
               tracklet->Update(*bbox, cur_frame_id_);
               // remove from new vec & push_back into active vec
               tracklet->MarkTrackState(kTrackState::Active);
               active_tracks_.push_back(new_tracks_[i]);
               new_tracks_.erase(new_tracks_.begin() + i);
          } else {
               // remove from new vec & push_back into removed vec
               tracklet->MarkTrackState(kTrackState::Removed);
               new_tracks_.erase(new_tracks_.begin() + i);
          }
     }
     for (int32_t i = 0; i < unmatched_high_conf_dets.size(); i++) {
          if (dets_assignment_res[i] < 0) {
               unmatched_twice_high_conf_dets.push_back(unmatched_high_conf_dets[i]);
          }
     }

     // 4.1 Mark this frame's final unmatched high-conf dets as new tracklets
     for (const auto& bbox : unmatched_twice_high_conf_dets) {
          if (bbox->cls_confidence < add_thresh_score_) continue;
          // new_tracks_.push_back(std::make_shared<Track>(*bbox, all_tracks_num_++, cur_frame_id_));
          std::shared_ptr<Track> ptr(new XywhTrack());
          ptr->Initialize(*bbox, all_tracks_num_++, cur_frame_id_);
          new_tracks_.push_back(ptr);
     }

     // 4.2 Remove the lost tracklets whose lost time is longer than the threshold
     for (auto it = lost_tracks_.begin(); it != lost_tracks_.end();) {
          if (cur_frame_id_ - it->get()->GetLastFrameId() > max_lost_frame_num_) {
               it->get()->MarkTrackState(kTrackState::Removed);
               it = lost_tracks_.erase(it);
          } else {
               it++;
          }
     }

     // for (auto it = new_tracks_.begin(); it != new_tracks_.end(); it++) {
     //      std::cout << it->use_count() << " ";
     // }
     // std::cout << std::endl;
     // for (auto it = active_tracks_.begin(); it != active_tracks_.end(); it++) {
     //      std::cout << it->use_count() << " ";
     // }
     // std::cout << std::endl;
     // for (auto it = lost_tracks_.begin(); it != lost_tracks_.end(); it++) {
     //      std::cout << it->use_count() << " ";
     // }
     // std::cout << std::endl;
     unmatched_boxes = unmatched_low_conf_dets;
     cur_frame_id_++;
}