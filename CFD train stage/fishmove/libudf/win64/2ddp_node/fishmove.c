#include "udf.h"
#include <math.h>
#define k 2.0*M_PI/0.95
#define w RP_Get_Real("w1")
#define A RP_Get_Real("a1")
#define mass 0.082122
#define INIT_V 0.0
static real dx = 0.0;
static real x = 0.0;
static real ve = 0.0;
static real a = 0.0;
static real y = 0.0;
DEFINE_GRID_MOTION(fish, domain, dt, time, dtime)
{
	Thread* tf = DT_THREAD(dt);
	face_t f;
	int n;
	Node* v;
	real xfish = 0.0;

	SET_DEFORMING_THREAD_FLAG(THREAD_T0(tf));
	begin_f_loop(f, tf)
	{
		f_node_loop(f, tf, n)
		{
			v = F_NODE(f, tf, n);
			if (NODE_POS_NEED_UPDATE(v))
			{
				NODE_POS_UPDATED(v);
				real P1 = time;
				real P2 = (1 - 1 / (1 + 10 * P1)) * (A - 0.08 * (NODE_X(v) - x) + 0.16 * (NODE_X(v) - x) * (NODE_X(v) - x)) * sin(k * (NODE_X(v) - x) - w * P1);
				P1 = time - dtime;
				real P3 = (1 - 1 / (1 + 10 * P1)) * (A - 0.08 * (NODE_X(v) - x) + 0.16 * (NODE_X(v) - x) * (NODE_X(v) - x)) * sin(k * (NODE_X(v) - x) - w * P1);
				NODE_Y(v) = NODE_Y(v) + (P2 - P3);
			}
		}
	}
	end_f_loop(f, tf);
	Message("Now is grid_motion\n");
}
DEFINE_EXECUTE_AT_END(execute_at_end)
{
	real f_glob[3] = { 0,0,0 };
	real m_glob[3] = { 0,0,0 };
	real x_cg[3] = { 0,0,0 };
	x_cg[0] = x;
	x_cg[1] = y;
	real dtime1 = CURRENT_TIMESTEP;
	real time1 = CURRENT_TIME;
#if RP_NODE
	if (!Data_Valid_P())
		return;
	Domain* domain1 = Get_Domain(1);  //return fluid domain
	Thread* tf1 = Lookup_Thread(domain1, 11);
	Compute_Force_And_Moment(domain1, tf1, x_cg, f_glob, m_glob, FALSE);
	real ve_before = ve, a_before = a;
	a = f_glob[0] / mass;
	ve = ve + (a + a_before) * dtime1 / 2;
	dx = (ve + ve_before) * dtime1 / 2;
	x = x + dx;
#endif
	node_to_host_real_4(a, ve, x, dx);
#if RP_HOST
	FILE* fpx = NULL;
	fpx = fopen("positionx.txt", "a");
	fprintf(fpx, "%.32f %.32f %.32f %.32f\n", time1,x,ve,a);
	fclose(fpx);
#endif
}
DEFINE_ZONE_MOTION(rotor, omega, axis, origin, velocity, time, dtime)
{
	Message("Now is zone_motion\n");
	*omega = 0;
	origin[0] = x;
	origin[1] = 0;
	velocity[1] = 0;
	velocity[0] = ve;
}
DEFINE_ON_DEMAND(assignment)
{
	real x_date[4] = { 0,0,0,0 };
	dx = x_date[3];
	x = x_date[0];
	ve = x_date[1];
	a = x_date[2];
	Message("dx=%f,x=%f,vx=%f,ax=%f \n", dx, x, ve, a);
}

DEFINE_PROFILE(velocity, thread, position)
{
	face_t face;
	real flow_time = CURRENT_TIME;
	static real velocity_offset = 0.0;
	static int last_update_time = 7.5;

	if (flow_time > last_update_time)
	{
		velocity_offset += 0.1;
		last_update_time+= 7.5;
	}

	begin_f_loop(face, thread)
	{
		F_PROFILE(face, thread, position) = INIT_V + velocity_offset;
	}
	end_f_loop(face, thread)
}


