#include <cassert>
#include <cmath>
#include <deque>
#include <utility>
#include <vector>

using namespace std;

class FITPS
{
public:
  FITPS(int cycle_size, int buffer_size, int thresh) : cycle_size(cycle_size), buffer_size(buffer_size), thresh(thresh)
  {
    assert(buffer_size > cycle_size);
    clear();
  }

  pair<vector<float>, vector<float>> add_samples(float volt_sample,
                                                 float amp_sample)
  {
    if (volt_buffer.size() == buffer_size)
    {
      zero_crossings[0] -= 1;
      zero_crossings[1] -= 1;
      volt_buffer.erase(volt_buffer.begin());
      amp_buffer.erase(amp_buffer.begin());
    }
    volt_buffer.emplace_back(volt_sample);
    amp_buffer.emplace_back(amp_sample);

    if (volt_buffer.size() > 1)
    {
      int prev_index = volt_buffer.size() - 2;
      float v_prev = volt_buffer[prev_index];
      float v_last = volt_buffer.back();

      if (v_prev < 0 && v_last >= 0)
      {
        // printf("%.4f\n", v_prev);
        if (zero_crossings.size() == 2)
        {
          zero_crossings.pop_front();
        }
        zero_crossings.emplace_back(prev_index);

        float shift = -v_prev / (v_last - v_prev + 1e-9);
        if (zero_crossing_shifts.size() == 2)
        {
          zero_crossing_shifts.pop_front();
        }
        zero_crossing_shifts.emplace_back(shift);
      }

      if (zero_crossings.size() == 2)
      {
        int actual_size = static_cast<int>(zero_crossings[1] - zero_crossings[0]);
        int deviation = std::abs(actual_size - cycle_size);

        if (deviation > thresh)
        {

          zero_crossings = {0};
          zero_crossing_shifts = {zero_crossing_shifts[1]};

          float i_1 = amp_buffer[prev_index];
          float i_2 = amp_buffer.back();

          clear();

          volt_buffer.emplace_back(v_prev);
          volt_buffer.emplace_back(v_last);

          amp_buffer.emplace_back(i_1);
          amp_buffer.emplace_back(i_2);

          return {};
        }

        vector<float> volts = allocate(volt_buffer);
        vector<float> amps = allocate(amp_buffer);

        zero_crossings = {0};
        zero_crossing_shifts = {zero_crossing_shifts[1]};

        float i_1 = amp_buffer[prev_index];
        float i_2 = amp_buffer.back();

        clear();

        volt_buffer.emplace_back(v_prev);
        volt_buffer.emplace_back(v_last);

        amp_buffer.emplace_back(i_1);
        amp_buffer.emplace_back(i_2);

        return {volts, amps};
      }
    }

    return {};
  }

  void clear()
  {
    volt_buffer.clear();
    volt_buffer.shrink_to_fit();
    volt_buffer.reserve(buffer_size);

    amp_buffer.clear();
    amp_buffer.shrink_to_fit();
    amp_buffer.reserve(buffer_size);
  }
  pair<vector<vector<float>>, vector<vector<float>>> transform(const vector<float> &volts,
                                                               const vector<float> &amps,
                                                               vector<int> locs = {})
  {
    vector<vector<float>> all_volt_cycles;
    vector<vector<float>> all_amp_cycles;

    if (!locs.empty() && find(locs.begin(), locs.end(), -1) == locs.end()) // Handling None in locs
    {
      // Convert locs to array-like structure
      vector<int> locs_array = locs;

      // Shift locs by minimum
      int locs_min = *min_element(locs_array.begin(), locs_array.end());
      for (auto &loc : locs_array)
      {
        loc -= locs_min;
      }

      // Clip locs
      int x_shape_prod = volts.size(); // Assuming same size as volts
      for (auto &loc : locs_array)
      {
        loc = max(0, min(loc, x_shape_prod - 1));
      }

      assert(*min_element(locs_array.begin(), locs_array.end()) >= 0);
      assert(*max_element(locs_array.begin(), locs_array.end()) < x_shape_prod);
    }

    for (size_t i = 0; i < volts.size(); ++i)
    {
      auto result = add_samples(volts[i], amps[i]);

      if (!result.first.empty() && !result.second.empty())
      {
        all_volt_cycles.push_back(result.first);
        all_amp_cycles.push_back(result.second);
      }
    }

    return {all_volt_cycles, all_amp_cycles};
  }

private:
  int cycle_size;
  int buffer_size;
  int thresh;
  vector<float> volt_buffer;
  vector<float> amp_buffer;
  deque<size_t> zero_crossings;
  deque<float> zero_crossing_shifts;

  vector<float> allocate(const vector<float> &buffer)
  {
    float real_start = zero_crossings[0] + zero_crossing_shifts[0];
    float real_end = zero_crossings[1] + zero_crossing_shifts[1];
    float real_len = real_end - real_start;

    float dist = real_len / cycle_size;

    vector<float> out(cycle_size, 0.0f);

    for (int k = 0; k < cycle_size; ++k)
    {
      float k1 = real_start + dist * k;
      int k2 = static_cast<int>(floor(k1));
      int k3 = static_cast<int>(ceil(k1));

      float value =
          buffer[k2] + (buffer[k3] - buffer[k2]) * zero_crossing_shifts[0];

      out[k] = value;
    }

    return out;
  }
};