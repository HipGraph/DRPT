#include "drpt_timer.h"

using namespace std;
using namespace std::chrono;

template<typename T>
auto dmrpt::Timer<T>::get_timelapse_in_milliseconds()
{
  return this->timelapse;
}

template<typename T>
 auto dmrpt::Timer<T>::get_timelapse_in_seconds()
{
  return this->timelapse/1000;
}

template<typename T>
void dmrpt::Timer<T>::record_start_time()
{
  this->start_timestamp = high_resolution_clock::now();
}

template<typename T>
void dmrpt::Timer<T>::record_end_time ()
{
  this->stop_timestamp = high_resolution_clock::now();
  this->timelapse = duration_cast<microseconds> (this->stop_timestamp - this->start_timestamp);
}


