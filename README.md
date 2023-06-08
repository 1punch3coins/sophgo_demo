# sophgo_demo
A very simple inference framework for sophgo BM1684 and BM1684X series tpus.
<br>There are two models supported, yolov5s & yolov8s trained on COCO and openvino_roadseg. Check them below.
<br>https://github.com/ultralytics/ultralytics
<br>https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001
<br>The table belows shows the speed performance under BM1684X.
|Model|Cpu_Pre_process|Tpu_Inference|Cpu_Post_Process|
|-------|---------------|-------------|----------------|
|Yolov8s|2.0ms|6.5ms|0~1ms|
|Roadseg|1.5ms|11.0ms|13.0ms|

The Tpu_inference time is fixed as the model size and pre_processed image size is fixed. The pre_preocessed time and post_processed time
is determined by the original input image size. Have fun!
