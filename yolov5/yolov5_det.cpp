#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

struct AffineMatrix {
  float value[6];
};

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
  if (argc < 4) return false;
  if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    auto net = std::string(argv[4]);
    if (net[0] == 'n') {
      gd = 0.33;
      gw = 0.25;
    } else if (net[0] == 's') {
      gd = 0.33;
      gw = 0.50;
    } else if (net[0] == 'm') {
      gd = 0.67;
      gw = 0.75;
    } else if (net[0] == 'l') {
      gd = 1.0;
      gw = 1.0;
    } else if (net[0] == 'x') {
      gd = 1.33;
      gw = 1.25;
    } else if (net[0] == 'c' && argc == 7) {
      gd = atof(argv[5]);
      gw = atof(argv[6]);
    } else {
      return false;
    }
    if (net.size() == 2 && net[1] == '6') {
      is_p6 = true;
    }
  } else if (std::string(argv[1]) == "-d" && argc == 4) {
    engine = std::string(argv[2]);
    img_dir = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer, float** cpu_output_buffer) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

  *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
  context.enqueue(batchsize, gpu_buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;
  if (is_p6) {
    engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  } else {
    engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
  }
  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

// class dict 
std::map<int, std::string> class_id_to_name = {
    {0, "person"},
    {1, "bicycle"},
    {2, "car"},
    {3, "motorbike"},
    {4, "aeroplane"},
    {5, "bus"},
    {6, "train"},
    {7, "truck"},
    {8, "boat"},
    {9, "traffic_Light"},
    {10, "fire_Hydrant"},
    {11, "stop_Sign"},
    {12, "parking_Meter"},
    {13, "bench"},
    {14, "bird"},
    {15, "cat"},
    {16, "dog"},
    {17, "horse"},
    {18, "sheep"},
    {19, "cow"},
    {20, "elephant"},
    {21, "bear"},
    {22, "zebra"},
    {23, "giraffe"},
    {24, "backpack"},
    {25, "umbrella"},
    {26, "handbag"},
    {27, "tie"},
    {28, "suitcase"},
    {29, "frisbee"},
    {30, "skis"},
    {31, "snowboard"},
    {32, "sports_Ball"},
    {33, "kite"},
    {34, "baseball_Bat"},
    {35, "baseball_Glove"},
    {36, "skateboard"},
    {37, "surfboard"},
    {38, "tennis_Racket"},
    {39, "bottle"},
    {40, "wine_Glass"},
    {41, "cup"},
    {42, "fork"},
    {43, "knife"},
    {44, "spoon"},
    {45, "bowl"},
    {46, "banana"},
    {47, "apple"},
    {48, "sandwich"},
    {49, "orange"},
    {50, "broccoli"},
    {51, "carrot"},
    {52, "hot_Dog"},
    {53, "pizza"},
    {54, "donut"},
    {55, "cake"},
    {56, "chair"},
    {57, "sofa"},
    {58, "pottedplant"},
    {59, "bed"},
    {60, "diningtable"},
    {61, "toilet"},
    {62, "tvmonitor"},
    {63, "laptop"},
    {64, "mouse"},
    {65, "remote"},
    {66, "keyboard"},
    {67, "cell_Phone"},
    {68, "microwave"},
    {69, "oven"},
    {70, "toaster"},
    {71, "sink"},
    {72, "refrigerator"},
    {73, "book"},
    {74, "clock"},
    {75, "vase"},
    {76, "scissors"},
    {77, "teddy_Bear"},
    {78, "hair_Drier"},
    {79, "toothbrush"}
};

// Function to get class name from class ID
std::string get_class_name(int class_id) {
    auto it = class_id_to_name.find(class_id);
    return (it != class_id_to_name.end()) ? it->second : "Unknown";
}

// Function to get file name without extension
std::string get_file_name_without_extension(const std::string& file_name) {
    size_t last_dot = file_name.find_last_of(".");
    if (last_dot != std::string::npos) {
        return file_name.substr(0, last_dot);
    }
    return file_name;
}

std::vector<AffineMatrix> calculate_d2s(std::vector<cv::Mat>& img_batch, int dst_width, int dst_height) {
    std::vector<AffineMatrix> d2s_matrices;

    for (size_t i = 0; i < img_batch.size(); i++) {
        int src_width = img_batch[i].cols;
        int src_height = img_batch[i].rows;

        AffineMatrix s2d, d2s;
        float scale = std::min(static_cast<float>(dst_height) / src_height, static_cast<float>(dst_width) / src_width);

        s2d.value[0] = scale;
        s2d.value[1] = 0;
        s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
        s2d.value[3] = 0;
        s2d.value[4] = scale;
        s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

        cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
        cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
        cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

        memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

        d2s_matrices.push_back(d2s);
    }

    return d2s_matrices;
}


void convert_to_standard_format_with_d2s(double x_center, double y_center, double width, double height, double& left, double& top, double& right, double& bottom, int w_image, int h_image, const AffineMatrix& d2s) {
    // Convert to corners in the scaled image space
    left = x_center - width / 2.0;
    top = y_center - height / 2.0;
    right = x_center + width / 2.0;
    bottom = y_center + height / 2.0;

    // Apply the d2s transformation
    double l = d2s.value[0] * left + d2s.value[1] * top + d2s.value[2];
    double t = d2s.value[3] * left + d2s.value[4] * top + d2s.value[5];
    double r = d2s.value[0] * right + d2s.value[1] * bottom + d2s.value[2];
    double b = d2s.value[3] * right + d2s.value[4] * bottom + d2s.value[5];

    left = l;
    top = t;
    right = r;
    bottom = b;
}

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  bool is_p6 = false;
  float gd = 0.0f, gw = 0.0f;
  std::string img_dir;

  if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  // Create a model using the API directly and serialize it to a file
  if (!wts_name.empty()) {
    serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
    return 0;
  }

  // Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* gpu_buffers[2];
  float* cpu_output_buffer = nullptr;
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

  // Read images from directory
  std::vector<std::string> file_names;
  if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    std::cerr << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  // batch predict
  for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
    // Get a batch of images
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;
    for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
      cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
      img_batch.push_back(img);
      img_name_batch.push_back(file_names[j]);
    }

    // Preprocess
    cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

    // Calculate d2s matrices for each image in the batch
    std::vector<AffineMatrix> d2s_matrices = calculate_d2s(img_batch, kInputW, kInputH);


    // Run inference
    auto start = std::chrono::system_clock::now();
    infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
    auto end = std::chrono::system_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "inference time: " << elapsed_time << "ms" << std::endl;
    float fps = 1000.0 / elapsed_time;  // Calculate FPS as 1000 milliseconds divided by elapsed time in milliseconds
    std::cout << "FPS: " << fps << std::endl;
    
    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    
    

    // Print bounding box coordinates
    for (size_t batch_idx = 0; batch_idx < img_batch.size(); ++batch_idx) {
        std::string output_folder = "/home/quandang246/project/SCS_SMART_CART_SYSTEM/Final_results/detection-results/"; // Path to your detection-results in mAP
        std::string output_file_name = output_folder + get_file_name_without_extension(img_name_batch[batch_idx]) + ".txt";
        std::ofstream output_file(output_file_name);    
        if (!output_file.is_open()) {
            std::cerr << "Failed to open the output file." << std::endl;
            return -1;
        }  
        
        std::cout << "Batch " << batch_idx << ":\n";
        
        for (size_t detection_idx = 0; detection_idx < res_batch[batch_idx].size(); ++detection_idx) {
            const Detection& detection = res_batch[batch_idx][detection_idx];
            
            double left, top, right, bottom;
            int w_image, h_image;
            w_image = img_batch[batch_idx].cols; 
            h_image = img_batch[batch_idx].rows;
            std::cout << "YOLO_format " << detection.bbox[0] << " " << detection.bbox[1] << " " << detection.bbox[2] << " " << detection.bbox[3] << "\n";
            
            convert_to_standard_format_with_d2s(detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3], left, top, right, bottom, w_image, h_image, d2s_matrices[batch_idx]);
            
            cv::rectangle(img_batch[batch_idx], cv::Point(int(left), int(top)), cv::Point(int(right), int(bottom)), cv::Scalar(0x27, 0xC1, 0x36), 2);

            std::cout << "  Detection " << detection_idx << ":\n";
            std::cout << "    Converted Bounding Box: (" << left << ", " << top << ", " << right << ", " << bottom << ")\n";
            std::cout << "    Confidence: " << detection.conf << "\n";
            std::cout << "    Class ID: " << detection.class_id << "\n";
            std::cout << "    Class Name: " << get_class_name(detection.class_id) << "\n";

            output_file << get_class_name(detection.class_id) << " " << detection.conf << " "
                        << int(left) << " " << int(top) << " " << int(right) << " " << int(bottom) << "\n";
      }
      //cv::imshow("show", img_batch[batch_idx]);
      //cv::waitKey(0); 
      std::cout << "Results written to: " << output_file_name << std::endl;
    }
  
    // Draw bounding boxes
    draw_bbox(img_batch, res_batch);

    // Save images
    std::string image_folder = "/home/quandang246/project/SCS_SMART_CART_SYSTEM/Final_results/images/";
    for (size_t j = 0; j < img_batch.size(); j++) {
      cv::imwrite(image_folder + img_name_batch[j], img_batch[j]);
    }
  }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(gpu_buffers[0]));
  CUDA_CHECK(cudaFree(gpu_buffers[1]));
  delete[] cpu_output_buffer;
  cuda_preprocess_destroy();
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Print histogram of the output distribution
  // std::cout << "\nOutput:\n\n";
  // for (unsigned int i = 0; i < kOutputSize; i++) {
  //   std::cout << prob[i] << ", ";
  //   if (i % 10 == 0) std::cout << std::endl;
  // }
  // std::cout << std::endl;

  return 0;
}

