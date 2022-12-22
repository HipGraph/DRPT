#include "drpt_timer.h"

using namespace std;
using namespace std::chrono;

dmrpt::Timer::Timer()
{
}

auto dmrpt::Timer::get_timelapse_in_milliseconds()
{
  return this->timelapse;
}

 dmrpt::Timer::get_timelapse_in_seconds()
{
  return this->timelapse/1000;
}

void dmrpt::Timer::record_start_time()
{
  this->start_timestamp = high_resolution_clock::now();
}

void dmrpt::Timer::record_end_time ()
{
  this->stop_timestamp = high_resolution_clock::now();
  this->timelapse = duration_cast<microseconds> (this->stop_timestamp - this->start_timestamp);
}


