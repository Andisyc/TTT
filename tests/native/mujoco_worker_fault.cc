#include <mujoco/mujoco.h>

extern "C" void hpd_fail_sensor(const mjModel*, mjData*, int) {
  mju_error("native worker failure sentinel");
}
