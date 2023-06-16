#ifndef DET_STRUCTS_
#define DET_STRUCTS_
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct Bbox2D {
    int32_t cls_id;
    std::string cls_name;
    float cls_confidence;

    int32_t x;
    int32_t y;
    int32_t h;
    int32_t w;

    Bbox2D():
        cls_id(0), cls_confidence(0), x(0), y(0), h(0), w(0)
    {}
    Bbox2D(int32_t cls_id_, float cls_confidence_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_id(cls_id_), cls_confidence(cls_confidence_), x(x_), y(y_), w(w_), h(h_)
    {}
    Bbox2D(int32_t cls_id_, std::string cls_name_, float cls_confidence_, int32_t x_, int32_t y_, int32_t w_, int32_t h_):
        cls_id(cls_id_), cls_name(cls_name_), cls_confidence(cls_confidence_), x(x_), y(y_), w(w_), h(h_)
    {}
};

struct Point2D {
    int32_t x;
    int32_t y;
    Point2D(int32_t x_, int32_t y_):
        x(x_), y(y_)
    {}
};

struct PointXYZRGB {
    float x;
    float y;
    float z;
    int32_t r;
    int32_t g;
    int32_t b;
};

// struct Lane2D {
//     std::vector<Point2D> lane_points;
// };
typedef std::vector<Point2D> Lane2D ;

struct SegObj {
    int32_t label;
    std::vector<Point2D> pt;
    PointXYZRGB _pt_data;
};

#pragma pack(push,1)
enum VIDEOTYPE
{
    USB,
    RTSP,
    SDK,
};
#pragma pack(pop)

#endif
