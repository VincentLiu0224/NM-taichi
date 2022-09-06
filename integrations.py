import numpy as np
import taichi as ti
import time


ti.init(arch=ti.gpu, device_memory_fraction=0.2)
ti.init(default_fp=ti.f32)

@ ti.kernel
def inti_main():
    i = 0.01
    s = 1.-i
    r = 0.
    u = ti.Vector([i, s, r])                         # initial conditions

    #uf = ti.Vector([1.])

    # this part is really important, 
    t = 0.                                           # end time
    tmax = 1000.                                       # start time
    dt = 0.1                                        # delta time
    dx = 1e-4                                        # dx is prepared but unused in this case
    leng = ti.cast(10000, dtype=ti.i32)               # length of iteration---tmax/dt, better define manually
                                                     # 'leng' should be change along with the shape of field

    for c in range(leng):
        u_new, t_new = runge_kutta_4(u, t, dt, dx)       # integration method, change manually---euler_f, euler_b, runge_kutta_4
        field[c, 0] = u_new
        u = ti.cast(u_new, dtype=ti.f32)
        t = ti.cast(t_new, dtype=ti.f32)


@ ti.func
def euler_f(u, t, dt, dx):
    u_new = u + f_hw(u, t, dx)*dt
    t_new = t + dt
    return u_new, t_new



@ ti.func
def euler_b(u, t, dt, dx):
    u_new = quasi_newton_eb(u, t, dt, dx)
    t_new = t + dt
    return u_new, t_new


@ ti.func
def quasi_newton_eb(u0, t0, dt, dx):
    u_old = ti.cast(u0, dtype=ti.f32)
    # u_new = ti.cast(u0, dtype=ti.f32)
    for i in range(500):
        # print(i)
        dudx = (root(u_old+dx, u0, t0+dt, dt, dx) - root(u_old, u0, t0+dt, dt, dx))/dx
        u_old = ti.cast(u_old - (root(u_old, u0, t0+dt, dt, dx)/dudx), dtype=ti.f32)
    return u_old

@ ti.func
def root(u1, u0, t, dt, dx):
    return u1 - u0 - dt*f_hw(u1, t, dx)


@ ti.func
def runge_kutta_4(u, t, dt, dx):
    k0 = dt*f_hw(u, t, dx)
    k1 = dt*f_hw(u+(k0/2.0), t+(dt/2.0), dx)
    k2 = dt*f_hw(u+(k1/2.0), t+(dt/2.0), dx)
    k3 = dt*f_hw(u+k2, t+dt, dx)
    return u+(k0+(2.0*k1)+(2.0*k2)+k3)/6.0, t+dt

@ ti.func
def f_hw(u, t, dx):
    beta = 0.5
    kappa = 1/3
    i_new = beta*u[1]*u[0]-kappa*u[0]
    s_new = -beta*u[0]*u[1]
    r_new = kappa*u[0]
    return ti.Vector([i_new, s_new, r_new])

@ ti.func
def f(u, t, dx):
    return 3*u



field = ti.Vector.field(3, dtype=ti.f32, shape=(10000, 1))          # the compilation of all data points,
                                                                   # in which the 'row' dimension must be 
                                                                   # the same as 'leng' parameter 
#print(field)
#field[2, 0] = ti.Vector([10, 20, 30])
#print(field)
start = time.time()
inti_main()
print(field)
end = time.time()


print(end-start)