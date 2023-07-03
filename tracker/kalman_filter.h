#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

template<size_t Row_num, size_t Col_num>
using EigenMatrix = Eigen::Matrix<float, Row_num, Col_num>;
template<size_t Row_num>
using EigenVector = Eigen::Matrix<float, Row_num, 1>;

template<size_t Dim_x, size_t Dim_z>
class KalmanFilter {
public:
    KalmanFilter()
    {}
    KalmanFilter(const EigenVector<Dim_x>& X, const EigenMatrix<Dim_x, Dim_x>& P):
        vec_state_mean_(X),
        mat_state_covariance_(P)
    {}
    KalmanFilter(const EigenVector<Dim_x>& X, const EigenMatrix<Dim_x, Dim_x>& P, const EigenMatrix<Dim_x, Dim_x>& F, const EigenMatrix<Dim_z, Dim_x>& H):
        vec_state_mean_(X),
        mat_state_covariance_(P),
        mat_motion_(F),
        mat_project_(H)
    {}

public:
    void Initialize();

    // X(t) = F * X(t-1); P(t) = F * P(t-1) * F.T + Q
    void Predict(const EigenMatrix<Dim_x, Dim_x>& F, const EigenMatrix<Dim_x, Dim_x>& Q) {
        vec_state_mean_ = F * vec_state_mean_;
        mat_state_covariance_ = F * mat_state_covariance_ * F.transpose() + Q;
    }
    void Predict(const EigenMatrix<Dim_x, Dim_x>& Q) {
        vec_state_mean_ = mat_motion_ * vec_state_mean_;
        mat_state_covariance_ = mat_motion_ * mat_state_covariance_ * mat_motion_.transpose() + Q;
    }
    void Predict() {
        vec_state_mean_ = mat_motion_ * vec_state_mean_;
        mat_state_covariance_ = mat_motion_ * mat_state_covariance_ * mat_motion_.transpose() + mat_envrionment_convariance_;
    }

    void Project(EigenVector<Dim_z>& proj_mean, EigenMatrix<Dim_z, Dim_z>& proj_cov) {
        proj_mean = mat_project_ * vec_state_mean_;
        proj_cov = mat_project_ * mat_state_covariance_ * mat_project_.transpose();
    }
    int32_t PointToStateDistance(const EigenVector<Dim_z>& meassurment) {
        EigenVector<Dim_z> proj_mean;
        EigenMatrix<Dim_z, Dim_z> proj_cov;
        Project(proj_mean, proj_cov);

        // cov = L*L.t : L = cov.llt().matrixL()
        // maha_dist = x.t*cov*x = x.t*(L*L.t)^-1.x ---> x.t*(L.t).inv*L.inv*x
        // ---> x.t*(L.inv).t*L.inv*x ---> (L.inv*x).t * L.inv.x : (L.inverse()*x).array().square().sum()
        EigenVector<Dim_z> delta_mean = proj_mean - meassurment;
        EigenMatrix<Dim_z, Dim_z> L = proj_cov.llt().matrixL();
        int32_t maha_distance = (L.inverse()*delta_mean).array().square().sum();
        return maha_distance;
    }

    // X(t) = X(t) + K * (Z(t) - H * X(t); P(t) = P(t) - K * H * P(t)
    void Update(const EigenMatrix<Dim_z, Dim_x> H, const EigenVector<Dim_z>& Z, const EigenMatrix<Dim_z, Dim_z>& R) {
        const EigenMatrix<Dim_x, Dim_x> I = EigenMatrix<Dim_x, Dim_x>::Identity();
        const EigenMatrix<Dim_z, Dim_z> S = H * mat_state_covariance_ * H.transpose() + R;
        // const EigenMatrix<Dim_x, Dim_z> K = mat_state_covariance_ * H.transpose() * S.inverse();
        const EigenMatrix<Dim_x, Dim_z> K = S.llt().solve(mat_state_covariance_ * H.transpose());
        vec_state_mean_ = vec_state_mean_ + K * (Z - (H * vec_state_mean_));
        mat_state_covariance_ = (I - K * H) * mat_state_covariance_;
    }
    void Update(const EigenVector<Dim_z>& Z, const EigenMatrix<Dim_z, Dim_z>& R) {
        const EigenMatrix<Dim_x, Dim_x> I = EigenMatrix<Dim_x, Dim_x>::Identity();
        const EigenMatrix<Dim_z, Dim_z> S = mat_project_ * mat_state_covariance_ * mat_project_.transpose() + R;
        // const EigenMatrix<Dim_x, Dim_z> K = S.llt().solve(mat_state_covariance_ * H.transpose());
        const EigenMatrix<Dim_x, Dim_z> K = mat_state_covariance_ * mat_project_.transpose() * S.inverse(); // The matrix's size is small, so use inverse() is okay
        vec_state_mean_ = vec_state_mean_ + K * (Z - (mat_project_ * vec_state_mean_));
        mat_state_covariance_ = (I - K * mat_project_) * mat_state_covariance_;
    }

public:
    const EigenVector<Dim_x> GetStateMean() const {
        return vec_state_mean_;
    }

private:
    EigenVector<Dim_x> vec_state_mean_;
    EigenMatrix<Dim_x, Dim_x> mat_state_covariance_;
    EigenMatrix<Dim_x, Dim_x> mat_motion_;
    EigenMatrix<Dim_x, Dim_x> mat_envrionment_convariance_;
    EigenMatrix<Dim_z, Dim_x> mat_project_;
    
};