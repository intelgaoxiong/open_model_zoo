// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <iomanip>
#include <openvino/openvino.hpp>
#include <gflags/gflags.h>
#include <utils/common.hpp>
#include <utils/slog.hpp>

namespace {
constexpr char h_msg[] = "show this help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char m_msg[] = "path to an .xml file with a trained model";
DEFINE_string(m, "", m_msg);

constexpr char i_msg[] = "path to an input WAV file";
DEFINE_string(i, "", i_msg);

constexpr char d_msg[] = "specify a device to infer on (the list of available devices is shown below). Default is CPU";
DEFINE_string(d, "CPU", d_msg);

constexpr char o_msg[] = "path to an output WAV file. Default is noise_suppression_demo_out.wav";
DEFINE_string(o, "noise_suppression_demo_out.wav", o_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[-h]               " << h_msg
                  << "\n\t[--help]           print help on all arguments"
                  << "\n\t -m <MODEL FILE>   " << m_msg
                  << '\n';
        showAvailableDevices();
        slog::info << ov::get_openvino_version() << slog::endl;
        exit(0);
    } if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
    slog::info << ov::get_openvino_version() << slog::endl;
}

struct RiffWaveHeader {
    unsigned int riff_tag; // "RIFF" string
    int riff_length;       // Total length
    unsigned int wave_tag; // "WAVE"
    unsigned int fmt_tag;  // "fmt " string (note space after 't')
    int fmt_length;        // Remaining length
    short data_format;     // Data format tag, 1 = PCM
    short num_of_channels; // Number of channels in file
    int sampling_freq;     // Sampling frequency
    int bytes_per_sec;     // Average bytes/sec
    short block_align;     // Block align
    short bits_per_sample;
    unsigned int data_tag; // "data" string
    int data_length;       // Raw data length
};

const unsigned int fourcc(const char c[4]) {
    return (c[3] << 24) | (c[2] << 16) | (c[1] << 8) | (c[0]);
}

void read_wav(const std::string& file_name, RiffWaveHeader& wave_header, std::vector<int16_t>& wave) {
    std::ifstream inp_wave(file_name, std::ios::in|std::ios::binary);
    if(!inp_wave.is_open())
        throw std::runtime_error("fail to open " + file_name);

    inp_wave.read((char*)&wave_header, sizeof(RiffWaveHeader));

    std::string error_msg = "";
    #define CHECK_IF(cond) if(cond){ error_msg = error_msg + #cond + ", "; }

    // make sure it is actually a RIFF file with WAVE
    CHECK_IF(wave_header.riff_tag != fourcc("RIFF"));
    CHECK_IF(wave_header.wave_tag != fourcc("WAVE"));
    CHECK_IF(wave_header.fmt_tag != fourcc("fmt "));
    // only PCM
    CHECK_IF(wave_header.data_format != 1);
    // only mono
    CHECK_IF(wave_header.num_of_channels != 1);
    // only 16 bit
    CHECK_IF(wave_header.bits_per_sample != 16);
    // make sure that data chunk follows file header
    CHECK_IF(wave_header.data_tag != fourcc("data"));
    #undef CHECK_IF

    if (!error_msg.empty()) {
        throw std::runtime_error(error_msg + "for '" + file_name + "' file.");
    }

    size_t wave_size = wave_header.data_length / sizeof(int16_t);
    wave.resize(wave_size);

    inp_wave.read((char*)&(wave.front()), wave_size * sizeof(int16_t));
}

void write_wav(const std::string& file_name, const RiffWaveHeader& wave_header, const std::vector<int16_t>& wave) {
    std::ofstream out_wave(file_name, std::ios::out|std::ios::binary);
    if(!out_wave.is_open())
        throw std::runtime_error("fail to open " + file_name);

    out_wave.write((char*)&wave_header, sizeof(RiffWaveHeader));
    out_wave.write((char*)&(wave.front()), wave.size() * sizeof(int16_t));
}
}  // namespace

void printInputAndOutputsInfoShort(const ov::CompiledModel& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& input : network.inputs()) {
        std::cout << "    " << input.get_any_name() << " (node: " << input.get_node()->get_friendly_name()
                  << ") : " << input.get_element_type() << " / " << ov::layout::get_layout(input).to_string()
                  << std::endl;
    }

    std::cout << "Network outputs:" << std::endl;
    for (auto&& output : network.outputs()) {
        std::string out_name = "***NO_NAME***";
        std::string node_name = "***NO_NAME***";

        // Workaround for "tensor has no name" issue
        try {
            out_name = output.get_any_name();
        } catch (const ov::Exception&) {
        }
        try {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        } catch (const ov::Exception&) {
        }

        std::cout << "    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
                  << ov::layout::get_layout(output).to_string() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);
    parse(argc, argv);
    ov::Core core;

    //import VPU blob
    std::string blobFile = FLAGS_m + "dns.blob";
    std::ifstream modelStream(blobFile, std::ios_base::binary | std::ios_base::in);
    if (!modelStream.is_open()) {
        throw std::runtime_error("Cannot open model file " + blobFile);
    }
    auto compiled_model = core.import_model(modelStream, "VPUX", {});
    modelStream.close();

    printInputAndOutputsInfoShort(compiled_model);

    ov::InferRequest infer_request = compiled_model.create_infer_request();

    //read input bianry
    std::string inputFile = FLAGS_m + "input.dat";
    const int nRows = 1081;
    const int nColumns = 256;
    float* pDat = new float[nRows * nColumns];
    std::ifstream indat;
    indat.open(inputFile);
    for (int i = 0; i < nRows * nColumns; i++)
    {
        indat >> pDat[i];
    }
    indat.close();

    // Prepare input
    // get size of network input (patch_size)
    std::string input_name("mixture");
    ov::Shape inp_shape = compiled_model.input(input_name).get_shape();
    size_t patch_size = inp_shape[1];
    std::cout << "patch size = " << patch_size << std::endl;

    std::vector<float> inp_wave_fp32;
    std::vector<float> out_wave_fp32;

    size_t iter = ((nRows*nColumns) / patch_size);
    size_t inp_size = patch_size * iter;
    std::cout << "iter = " << iter << ", inp_size = " << inp_size << std::endl;
    inp_wave_fp32.resize(inp_size, 0);
    out_wave_fp32.resize(inp_size, 0);
    for (size_t i = 0; i < inp_size; ++i) {
        inp_wave_fp32[i] = pDat[i];
    }

    //State API implementation for VPUX
    std::vector<ov::VariableState> state_vector = infer_request.query_state();
    for (int i = 0; i < state_vector.size(); i++)
    {
        std::cout << "state_vector " << i << " = " << state_vector[i].get_name() << std::endl;
        auto& t = state_vector[i].get_state();
        std::cout << "   shape = " << t.get_shape() << ", element_type = " << t.get_element_type() << std::endl;

    }

    // initialize memory state before starting
    for (auto&& state : infer_request.query_state()) {
        state.reset();
    }

    auto start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iter; ++i) {
        ov::Tensor input_tensor(ov::element::f32, inp_shape, &inp_wave_fp32[i * patch_size]);
        infer_request.set_tensor(input_name, input_tensor);

        infer_request.infer();

        {
            // process output
            float* src = infer_request.get_tensor("325").data<float>();
            float* dst = &out_wave_fp32[i * patch_size];
            std::memcpy(dst, src, patch_size * sizeof(float));
        }
    } // for iter

    std::ofstream outdat;
    outdat.open("output.dat");
    outdat.setf(std::ios::fixed);
    for (int r = 0; r < nRows; r++)
    {
        for (int c = 0; c < nColumns; c++)
        {
            outdat << std::setprecision(10) << out_wave_fp32[r * nColumns + c] << " ";
        }
        outdat << std::endl;
    }
    outdat.close();

    return 0;
}
