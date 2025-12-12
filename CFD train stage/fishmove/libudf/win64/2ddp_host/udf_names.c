/* This file generated automatically. */
/*          Do not modify.            */
#include "udf.h"
#include "prop.h"
#include "dpm.h"
extern DEFINE_GRID_MOTION(fish, domain, dt, time, dtime);
extern DEFINE_EXECUTE_AT_END(execute_at_end);
extern DEFINE_ZONE_MOTION(rotor, omega, axis, origin, velocity, time, dtime);
extern DEFINE_ON_DEMAND(assignment);
extern DEFINE_PROFILE(velocity, thread, position);
__declspec(dllexport) UDF_Data udf_data[] = {
{"fish", (void(*)())fish, UDF_TYPE_GRID_MOTION},
{"execute_at_end", (void(*)())execute_at_end, UDF_TYPE_EXECUTE_AT_END},
{"rotor", (void(*)())rotor, UDF_TYPE_ZONE_MOTION},
{"assignment", (void(*)())assignment, UDF_TYPE_ON_DEMAND},
{"velocity", (void(*)())velocity, UDF_TYPE_PROFILE},
};
__declspec(dllexport) int n_udf_data = sizeof(udf_data)/sizeof(UDF_Data);
#include "version.h"
__declspec(dllexport) void UDF_Inquire_Release(int *major, int *minor, int *revision)
{
  *major = RampantReleaseMajor;
  *minor = RampantReleaseMinor;
  *revision = RampantReleaseRevision;
}
