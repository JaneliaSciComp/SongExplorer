/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*

Tool to create accuracy statistics from running an audio recognition model on a
continuous stream of samples.

This is designed to be an environment for running experiments on new models and
settings to understand the effects they will have in a real application. You
need to supply it with a long audio file containing sounds you want to recognize
and a text file listing the labels of each sound along with the time they occur.
With this information, and a frozen model, the tool will process the audio
stream, apply the model, and keep track of how many mistakes and successes the
model achieved.

The matched percentage is the number of sounds that were correctly classified,
as a percentage of the total number of sounds listed in the ground truth file.
A correct classification is when the right label is chosen within a short time
of the expected ground truth, where the time tolerance is controlled by the
'time_tolerance_ms' command line flag.

The wrong percentage is how many sounds triggered a detection (the classifier
figured out it wasn't silence or background noise), but the detected class was
wrong. This is also a percentage of the total number of ground truth sounds.

The false positive percentage is how many sounds were detected when there was
only silence or background noise. This is also expressed as a percentage of the
total number of ground truth sounds, though since it can be large it may go
above 100%.

The easiest way to get an audio file and labels to test with is by using the
'generate_streaming_test_wav' script. This will synthesize a test file with
randomly placed sounds and background noise, and output a text file with the
ground truth.

If you want to test natural data, you need to use a .wav with the same sample
rate as your model (often 16,000 samples per second), and note down where the
sounds occur in time. Save this information out as a comma-separated text file,
where the first column is the label and the second is the time in seconds from
the start of the file that it occurs.

Here's an example of how to run the tool:

bazel run tensorflow/examples/speech_commands:test_streaming_accuracy -- \
--wav=/tmp/streaming_test_bg.wav \
--graph=/tmp/conv_frozen.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--ground_truth=/tmp/streaming_test_labels.txt --verbose \
--clip_duration_ms=1000 --detection_threshold=0.70 --average_window_ms=500 \
--suppression_ms=500 --time_tolerance_ms=1500

 */

#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <vector>
#include <list>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/examples/speech_commands_custom/accuracy_utils.h"
#include "tensorflow/examples/speech_commands_custom/recognize_commands.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::string;
using tensorflow::uint16;
using tensorflow::uint32;

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session, int ncores) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions opts;
  opts.config.mutable_gpu_options()->set_allow_growth(true);
  opts.config.set_inter_op_parallelism_threads(ncores);
  opts.config.set_intra_op_parallelism_threads(ncores);
  session->reset(tensorflow::NewSession(opts));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file '", file_name,
                                        "' not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  return Status::OK();
}

}  // namespace

int main(int argc, char* argv[]) {
  string wav = "";
  string graph = "";
  string labels = "";
  string ground_truth = "";
  string input_data_name = "decoded_sample_data:0";
  string input_rate_name = "decoded_sample_data:1";
  string output_names = "labels_softmax";
  int nstrides = 1;
  float clip_duration_ms = 1000;
  float clip_stride_ms = 30;
  int ncores = 0;
  bool verbose = false;
  std::vector<Flag> flag_list = {
      Flag("wav", &wav, "audio file to be identified"),
      Flag("graph", &graph, "model to be executed"),
      Flag("labels", &labels, "path to file containing labels"),
      Flag("ground_truth", &ground_truth,
           "path to file containing correct times and labels of words in the "
           "audio as <word>,<timestamp in ms> lines"),
      Flag("input_data_name", &input_data_name,
           "name of input data node in model"),
      Flag("input_rate_name", &input_rate_name,
           "name of input sample rate node in model"),
      Flag("output_names", &output_names, "comma-separated names of output nodes in model"),
      Flag("nstrides", &nstrides, "how many context windows to process in parallel"),
      Flag("clip_duration_ms", &clip_duration_ms,
           "length of recognition window"),
      Flag("clip_stride_ms", &clip_stride_ms, "how often to run recognition"),
      Flag("ncores", &ncores, "how many CPU cores to use"),
      Flag("verbose", &verbose, "whether to log extra debugging information"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph, &session, ncores);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector<string> labels_list;
  Status read_labels_status = ReadLabelsFile(labels, &labels_list);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  std::vector<std::pair<string, tensorflow::int64>> ground_truth_list;
  Status read_ground_truth_status =
      tensorflow::ReadGroundTruthFile(ground_truth, &ground_truth_list);
  if (!read_ground_truth_status.ok()) {
    LOG(ERROR) << read_ground_truth_status;
    return -1;
  }

  string wav_string;
  Status read_wav_status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), wav, &wav_string);
  if (!read_wav_status.ok()) {
    LOG(ERROR) << read_wav_status;
    return -1;
  }
  std::vector<float> audio_data;
  uint32 sample_count;
  uint16 channel_count;
  uint32 sample_rate;
  Status decode_wav_status = tensorflow::wav::DecodeLin16WaveAsFloatVector(
      wav_string, &audio_data, &sample_count, &channel_count, &sample_rate);
  if (!decode_wav_status.ok()) {
    LOG(ERROR) << decode_wav_status;
    return -1;
  }
  if (channel_count != 1) {
    LOG(ERROR) << "Only mono .wav files can be used, but input has "
               << channel_count << " channels.";
    return -1;
  }

  const int64 clip_duration_samples = (clip_duration_ms * sample_rate) / 1000;
  const int64 clip_stride_samples = (clip_stride_ms * sample_rate) / 1000;
  Tensor audio_data_tensor(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({clip_duration_samples, 1}));

  Tensor sample_rate_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  sample_rate_tensor.scalar<int32>()() = sample_rate;

  std::vector<string> output_names_vec;
  std::stringstream ss(output_names);
  std::string item;
  while (std::getline(ss, item, ',')) {
      output_names_vec.push_back(item); }

  const int64 audio_data_end = (sample_count - clip_duration_samples);
  for (int64 audio_data_offset = 0; audio_data_offset < audio_data_end;
       audio_data_offset += clip_stride_samples) {
    const float* input_start = &(audio_data[audio_data_offset]);
    const float* input_end = input_start + clip_duration_samples;
    std::copy(input_start, input_end, audio_data_tensor.flat<float>().data());

    // Actually run the audio through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_data_name, audio_data_tensor},
                                      {input_rate_name, sample_rate_tensor}},
                                     {output_names_vec}, {}, &outputs);

    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }

    const int64 current_time_ms = (audio_data_offset * 1000) / sample_rate;

    std::vector<string>::iterator output_name;
    int i=0;
    if (verbose)
      for (output_name=output_names_vec.begin();
           output_name!=output_names_vec.end();
           output_name++) {
        LOG(INFO) << current_time_ms << "ms: " << *output_name << " " << outputs[i].SummarizeValue(outputs[i].NumElements());
        i++;
      }
  }

  return 0;
}