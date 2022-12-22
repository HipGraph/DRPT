#ifndef DISTRIBUTED_DRPT_TIMER_H
#define DISTRIBUTED_DRPT_TIMER_H

#include <chrono>

using namespace std;
using namespace std::chrono;

namespace dmrpt
{
template <class T, class Y>
class Timer {
 private:
  T start_timestamp;
  T stop_timestamp ;
  Y timelapse;

 public:
  void record_start_time ();
  void record_end_time ();
  Y get_timelapse_in_seconds ();
  Y get_timelapse_in_milliseconds ();
};

}

#endif //DISTRIBUTED_DRPT_TIMER_H
