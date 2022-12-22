#ifndef DISTRIBUTED_DRPT_TIMER_H
#define DISTRIBUTED_DRPT_TIMER_H

#include <chrono>

using namespace std;
using namespace std::chrono;

namespace dmrpt
{
class Timer {
 private:
  auto start_timestamp;
  auto stop_timestamp;
  auto timelapse;

 public:
  void record_start_time ();
  void record_end_time ();
  auto get_timelapse_in_seconds ();
  auto get_timelapse_in_milliseconds ();
};

}
