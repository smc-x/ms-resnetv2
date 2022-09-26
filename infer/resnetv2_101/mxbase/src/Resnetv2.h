#include <string>
#include <vector>
#include <memory>
#ifndef MxBase_ALEXNET_H
#define MxBase_ALEXNET_H
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/postprocess/include/ClassPostProcessors/Resnet50PostProcess.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"


struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    uint32_t classNum;
    uint32_t topk;
    bool softmax;
    bool checkTensor;
    std::string modelPath;
};

struct ImageShape {
    uint32_t width;
    uint32_t height;
};

class Resnetv2 {
 public:
     APP_ERROR Init(const InitParam &initParam);
     APP_ERROR DeInit();
     APP_ERROR ReadImage(const std::string &imgPath, cv::Mat *imageMat);
     APP_ERROR ResizeImage(cv::Mat *imageMat);
     APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
     APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
     APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                              std::vector<std::vector<MxBase::ClassInfo>> *clsInfos);
     APP_ERROR Process(const std::string &imgPath, const std::string &resPath);
     double GetInferCostMilliSec() const {return inferCostTimeMilliSec;}
 private:
     APP_ERROR SaveResult(const std::string &imgPath, const std::string &resPath,
            const std::vector<std::vector<MxBase::ClassInfo>> &batchClsInfos);
 private:
     std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
     std::shared_ptr<MxBase::Resnet50PostProcess> post_;
     MxBase::ModelDesc modelDesc_;
     uint32_t deviceId_ = 0;
     double inferCostTimeMilliSec = 0.0;
};
#endif
