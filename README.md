# sophgo_demo
A very simple inference framework for sophgo BM1684 and BM1684X series tpus.
<br>There are three models supported, yolov5s & yolov8s(det+seg) trained on COCO, openvino_roadseg and uf_landetv2 trained on CuLane. Check them below.
<br>https://github.com/ultralytics/ultralytics
<br>https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001
<br>https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
<br>The table below shows the speed performance under BM1684X in FP16 mode. The test image size is 1920x1080.
|Model|Tpu_Pre_process|Tpu_Inference|Cpu_Post_Process|
|-------|---------------|-------------|----------------|
|Yolov8s|2.1ms|6.5ms|0~1ms|
|Yolov8s_seg|2.6ms|12.4ms|1.6ms|
|Roadseg|2.1ms|11.0ms|6.0ms|
|Uf_landetv2|2.6ms|15.4ms|1.1ms|

The Tpu_inference time is fixed as the model's input size is fixed. The pre_preocessed time and post_processed time
is determined by the original input image's size. Have fun!


https://github.com/1punch3coins/sophgo_demo/assets/133486152/458672e0-b405-4894-babb-07489ba8ee5f

