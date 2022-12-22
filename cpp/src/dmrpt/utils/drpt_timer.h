#ifndef DISTRIBUTED_DRPT_TIMER_H
#define DISTRIBUTED_DRPT_TIMER_H

#include <chrono>

using namespace std;
using namespace std::chrono;

namespace dmrpt
{
template <class T>
class Timer {
 private:
  T start_timestamp;
  T stop_timestamp ;
  T timelapse;

 public:
  void record_start_time ();
  void record_end_time ();
  T get_timelapse_in_seconds ();
  T get_timelapse_in_milliseconds ();
};

}

#endif //DISTRIBUTED_DRPT_TIMER_H
