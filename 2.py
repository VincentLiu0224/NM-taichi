#from cProfile import run
#from imp import init_builtin
from cProfile import run
import numpy as np
import taichi as ti
import time


ti.init(arch=ti.cpu, default_fp=ti.f32, dynamic_index=True)


@ ti.kernel
def inti_main():
    t = 0.                                           # end time
    tmax = 0.01                                       # start time
    dt = ti.cast(0.0000001,dtype=ti.f32)                                        # delta time
    dx = ti.cast(2/31, dtype=ti.f32)                                        # dx is prepared but unused in this case
    leng = ti.cast(100000, dtype=ti.i32)               # length of iteration---tmax/dt, better define manually
                                                     # 'leng' should be change along with the shape of field
    u = init_con
    ti.loop_config(serialize=True)
    for c in range(100000):
        u_new, t_new = runge_kutta_4(u, t, dt, dx)       # integration method, change manually---euler_f, runge_kutta_4
        field[c, 0] = u_new
        u = ti.cast(u_new, dtype=ti.f32)
        t = ti.cast(t_new, dtype=ti.f32)


@ ti.func
def euler_f(u, t, dt, dx):
    u_new = u + heat_equ(u, t, dx)*dt
    t_new = t + dt
    # print(t)
    return u_new, t_new




@ ti.func
def runge_kutta_4(u, t, dt, dx):
    k0 = heat_equ(u, t, dx)
    k1 = heat_equ(u+(dt*k0/2.0), t+(dt/2.0), dx)
    k2 = heat_equ(u+(dt*k1/2.0), t+(dt/2.0), dx)
    k3 = heat_equ(u+dt*k2, t+dt, dx)
    return u + dt*(k0+(2.0*k1)+(2.0*k2)+k3)/6.0, t+dt

 

@ ti.func
def d2udx2(u, dx):
    d2u = ti.cast(dut, dtype=ti.f32)
    for i in range(1, 31):
        d2u[i] = (u[i+1]-2.*u[i]+u[i-1])/(dx**2.)
    #print(d2u)
    return d2u


@ ti.func
def dudx(u, dx):                                                 # central diff method
    du = dut
    for i in range(1, 31):
        du[i-1] = (u[i+1]-u[i-1])/(dx*2.)
    return du


@ ti.func
def heat_equ(u, t, dx, k=130.):
    res = k*d2udx2(u, dx)
    return res




field = ti.Vector.field(32, dtype=ti.f32, shape=(100000, 1))          # the compilation of all data points,
                                                                   # in which the 'row' dimension must be 
                                                                   # the same as 'leng' parameter 
ic = np.zeros(32)
ic[1:-1] = np.ones(30) 
init_con = ti.Vector(list(ic), dt=ti.f32)                   # global initial condition
d = ti.types.vector(32, ti.f32)                             # global derivatives template
dut = d(np.zeros(32))
print(init_con)
start = time.time()
inti_main()
end = time.time()

#result = np.array(field)
print(field[99999, 0])

#inti_main()
print(end-start)



