// Standalone, single-threaded ARMv7 inference probe; no audio callback.
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

template<class T> std::vector<T> read_binary(const std::string& path, size_t count) {
  std::ifstream input(path, std::ios::binary);
  std::vector<T> values(count);
  if (!input.read(reinterpret_cast<char*>(values.data()), count*sizeof(T)))
    throw std::runtime_error("Missing/truncated fixture: " + path);
  return values;
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: probe model.onnx results-dir");
    static_assert(sizeof(void*) == 4, "This probe must be built for ARM32");
    const std::string dir = argv[2];
    int steps, width, layers, hidden, eos;
    std::ifstream fixture(dir + "/fixture.txt");
    if (!(fixture >> steps >> width >> layers >> hidden >> eos) || steps != 256 ||
        width <= 0 || width > 10000 || layers != 2 || hidden != 1024)
      throw std::runtime_error("Invalid fixture dimensions");
    std::vector<int64_t> tokens(steps);
    for (auto& id : tokens) if (!(fixture >> id)) throw std::runtime_error("Missing token");
    const auto expected = read_binary<float>(dir + "/expected-logits.f32", size_t(steps)*width);
    const size_t state_size = size_t(layers)*hidden;
    const auto expected_state = read_binary<float>(dir + "/expected-state.f32", 2*state_size);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nanomaestro-arm32");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, argv[1], options);
    const std::array<int64_t,2> input_shape{1,1};
    const std::array<int64_t,3> state_shape{layers,1,hidden};
    std::vector<float> h(state_size,0), c(state_size,0);
    const char* input_names[] = {"input","h","c"};
    const char* output_names[] = {"logits","h_out","c_out"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto infer = [&](int64_t token) {
      std::array<Ort::Value,3> values{
        Ort::Value::CreateTensor<int64_t>(memory,&token,1,input_shape.data(),2),
        Ort::Value::CreateTensor<float>(memory,h.data(),h.size(),state_shape.data(),3),
        Ort::Value::CreateTensor<float>(memory,c.data(),c.size(),state_shape.data(),3)};
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,values.data(),3,output_names,3);
      const size_t counts[] = {size_t(width),state_size,state_size};
      for (int j=0;j<3;++j) {
        if (outputs[j].GetTensorTypeAndShapeInfo().GetElementCount()!=counts[j])
          throw std::runtime_error("Unexpected output shape");
        const float* p=outputs[j].GetTensorData<float>();
        if (!std::all_of(p,p+counts[j],[](float x){return std::isfinite(x);}))
          throw std::runtime_error("Non-finite neural output");
      }
      std::copy_n(outputs[1].GetTensorData<float>(),state_size,h.begin());
      std::copy_n(outputs[2].GetTensorData<float>(),state_size,c.begin());
      const auto* logits=outputs[0].GetTensorData<float>();
      return std::vector<float>(logits,logits+width);
    };
    double max_logit_error=0, max_probability_error=0;
    int argmax_matches=0;
    const auto started=std::chrono::steady_clock::now();
    for (int i=0;i<steps;++i) {
      const auto actual=infer(tokens[i]);
      const float* reference=expected.data()+size_t(i)*width;
      const auto best=std::max_element(actual.begin(),actual.end())-actual.begin();
      const auto ref_best=std::max_element(reference,reference+width)-reference;
      argmax_matches += best==ref_best;
      const double amax=actual[best], rmax=reference[ref_best];
      double asum=0,rsum=0;
      for(int j=0;j<width;++j){asum+=std::exp(actual[j]-amax);rsum+=std::exp(reference[j]-rmax);}
      for(int j=0;j<width;++j) {
        max_logit_error=std::max(max_logit_error,double(std::abs(actual[j]-reference[j])));
        max_probability_error=std::max(max_probability_error,
          std::abs(std::exp(actual[j]-amax)/asum-std::exp(reference[j]-rmax)/rsum));
      }
    }
    double state_error=0;
    for(size_t j=0;j<state_size;++j) {
      state_error=std::max(state_error,double(std::abs(h[j]-expected_state[j])));
      state_error=std::max(state_error,double(std::abs(c[j]-expected_state[j+state_size])));
    }
    // A second trajectory is sampled entirely by the ARM process.
    std::fill(h.begin(),h.end(),0); std::fill(c.begin(),c.end(),0);
    std::mt19937 random(20261121);
    std::ofstream sampled(dir+"/arm32-token-ids.txt");
    int64_t token=tokens[0];
    for(int i=0;i<512;++i) {
      auto logits=infer(token);
      if(i<4) token=tokens[i+1];
      else {
        for(auto& value:logits)value/=.9f;
        if(eos>=0&&eos<width)logits[eos]-=1;
        std::vector<int> ids(width);std::iota(ids.begin(),ids.end(),0);
        const int k=std::min(32,width);
        std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),
          [&](int a,int b){return logits[a]>logits[b];});
        std::vector<double> p(k);
        for(int j=0;j<k;++j)p[j]=std::exp(logits[ids[j]]-logits[ids[0]]);
        std::discrete_distribution<int> distribution(p.begin(),p.end());
        token=ids[distribution(random)];
      }
      sampled<<token<<' ';
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    const bool matches=max_probability_error<=.02 && argmax_matches>=steps*.9;
    std::ofstream report(dir+"/arm32-report.json");
    report<<"{\n  \"pointer_bits\":32,\n  \"runtime\":\""<<OrtGetApiBase()->GetVersionString()
      <<"\",\n  \"teacher_forced_steps\":"<<steps
      <<",\n  \"sampled_steps\":512,\n  \"argmax_matches\":"<<argmax_matches
      <<",\n  \"max_logit_error\":"<<max_logit_error
      <<",\n  \"max_probability_error\":"<<max_probability_error
      <<",\n  \"max_state_error\":"<<state_error
      <<",\n  \"emulated_elapsed_seconds\":"<<elapsed
      <<",\n  \"emulated_process_maxrss_kib\":"<<usage.ru_maxrss
      <<",\n  \"reference_check_passed\":"<<(matches?"true":"false")
      <<",\n  \"hardware_benchmark\":false\n}\n";
    std::cout<<"ARM32: "<<steps<<" oracle steps, 512 sampled steps; argmax "
      <<argmax_matches<<"/"<<steps<<"; max probability error "<<max_probability_error<<"\n";
    if(!matches)throw std::runtime_error("ARM32/reference deviation exceeds declared tolerance");
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<"\n";return 1; }
}
