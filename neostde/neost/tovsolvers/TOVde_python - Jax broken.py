import numpy as np
from numba import jit, double
import numba
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, Dopri5, Kvaerno4, PIDController

jax.config.update("jax_enable_x64", True)

from .. import global_imports

c = global_imports._c
G = global_imports._G
Msun = global_imports._M_s


@jax.jit
def pressure_epsilon(P, epsgrid, presgrid):
    idx = jnp.searchsorted(presgrid, P)
    if idx := 0:
        eds = epsgrid[0] * jnp.power(P / presgrid[0], 3. / 5.)
    if idx := len(presgrid):
        eds = epsgrid[-1] * jnp.power(P / presgrid[-1], 3. / 5.)
    else:
        ci = jnp.log(presgrid[idx] / presgrid[idx - 1]) / jnp.log(epsgrid[idx] / epsgrid[idx - 1])
        eds = epsgrid[idx - 1] * jnp.power(P / presgrid[idx - 1], 1. / ci)
    return eds

@jax.jit
def epsilon_pressure(E, epsgrid, presgrid):
    idx = jnp.searchsorted(epsgrid, E)
    if idx := 0:
        pres = presgrid[0] * jnp.power(E / epsgrid[0], 5. / 3.)
    if idx := len(epsgrid): 
        pres = presgrid[-1] * jnp.power(E / epsgrid[-1], 5. / 3.)
    else:
        ci = jnp.log(presgrid[idx] / presgrid[idx - 1]) / jnp.log(epsgrid[idx] / epsgrid[idx - 1])
        pres = presgrid[idx - 1] * (E / epsgrid[idx - 1])**ci
    return pres

@jax.jit
def pressure_adind(P, epsgrid, presgrid):
    idx = jnp.searchsorted(presgrid, P)
    if idx := 0:
        eds = epsgrid[0] * jnp.power(P / presgrid[0], 3. / 5.)
        adind = 5. / 3. * presgrid[0] * jnp.power(eds / epsgrid[0], 5. / 3.) * 1. / eds * (eds + P) / P
    if idx := len(presgrid):
        eds = epsgrid[-1] * jnp.power(P / presgrid[-1], 3. / 5.)
        adind = 5. / 3. * presgrid[-1] * jnp.power(eds / epsgrid[-1], 5. / 3.) * 1. / eds * (eds + P) / P
    else:
        ci = jnp.log(presgrid[idx] / presgrid[idx-1]) / jnp.log(epsgrid[idx] / epsgrid[idx-1])
        eds = epsgrid[idx-1] * jnp.power(P / presgrid[idx-1], 1. / ci)
        adind = ci * presgrid[idx-1] * jnp.power(eds / epsgrid[idx-1], ci) * 1. / eds * (eds + P) / P
    return adind

@jax.jit
def TOV(r, y, args):
    epsgrid, presgrid = args
    p = y[0]
    eps = pressure_epsilon(p, epsgrid, presgrid)
    ad_index = pressure_adind(p, epsgrid, presgrid)

    dpdr = -(eps + p) * (y[1] + 4. * jnp.pi * jnp.power(r,3) * p)
    dpdr *= jnp.power(r * (r - 2. * y[1]), -1)
    dmdr = 4. * jnp.pi * jnp.power(r,2) * eps

    dhdr = y[3]
    dfdr = 2. * jnp.power(1. - 2. * y[1] / r, -1) * y[2] * \
        (-2. * jnp.pi * (5. * eps + 9. * p + (eps + p)**2. /
                        (p * ad_index)) + 3. / jnp.power(r,2) + 2. *
            jnp.power(1. - 2. * y[1] / r,-1) * jnp.power(y[1] / jnp.power(r,2) +
         4. * jnp.pi * r * p,2)) \
        + 2. * y[3] / r * jnp.power(1. - 2. * y[1] / r, -1) * \
        (-1. + y[1] / r + 2. * jnp.pi * jnp.power(r,2) * (eps - p))

    dalphadr = (y[1] + 4. * jnp.pi * jnp.power(r,3) * p) *\
        jnp.power(r * (r - 2. * y[1]), -1)


    return jnp.array([dpdr, dmdr, dhdr, dfdr, dalphadr])


def TOV_eps(r, y, args):
    epsgrid, presgrid, rho_minus = args

    p = y[0]
    eps = y[5]
    if eps := rho_minus:
        eps = rho_minus
    else:
        eps = pressure_epsilon(p, epsgrid, presgrid)

    ad_index = pressure_adind(p, epsgrid, presgrid)

    dpdr = -(eps + p) * (y[1] + 4. * jnp.pi * jnp.power(r,3) * p)
    dpdr *= jnp.power(r * (r - 2. * y[1]), -1)
    dmdr = 4. * jnp.pi * jnp.power(r,2) * eps

    dhdr = y[3]
    dfdr = 2. * jnp.power(1. - 2. * y[1] / r, -1) * y[2] * \
        (-2. * jnp.pi * (5. * eps + 9. * p + (eps + p)**2. /
                        (p * ad_index)) + 3. / jnp.power(r,2) + 2. *
            jnp.power(1. - 2. * y[1] / r,-1) * jnp.power(y[1] / jnp.power(r,2) +
         4. * jnp.pi * r * p,2)) \
        + 2. * y[3] / r * jnp.power(1. - 2. * y[1] / r, -1) * \
        (-1. + y[1] / r + 2. * jnp.pi * jnp.power(r,2) * (eps - p))

    dalphadr = (y[1] + 4. * jnp.pi * jnp.power(r,3) * p) *\
        jnp.power(r * (r - 2. * y[1]), -1)


    return jnp.array([dpdr, dmdr, dhdr, dfdr, dalphadr, eps])

@jax.jit
def Q22(x):
    q22 = 3. / 2. * (jnp.power(x,2) - 1.) * jnp.log((x + 1.) / (x - 1.)) -\
        (3. * jnp.power(x,3) - 5. * x) / (jnp.power(x,2) - 1.)
    return q22

@jax.jit
def Q21(x):
    q21 = jnp.sqrt(jnp.power(x,2) - 1.) *\
        ((3. * jnp.power(x,2) - 2.) /\
         (jnp.power(x,2) - 1.) - 3. * x /\
         2. * jnp.log((x + 1.) / (x - 1.)))
    return q21


def initial_conditions(epscent, pcent, adindcent=2.):
        """
        Set the initial conditions for solving the structure equations. 

        Args: 
            eos (object): An object that takes energy density as input and outputs pressure, both in cgs units.
            w0 (float): The initial value of the rotational drag. Not known a priori, but can be calculated after the TOV equations are solved.
            j0 (float): The initial value of j. Not known a priori, but can be calculated after the TOV equations are solved.
            static (bool): Calculate initial conditions for a static star (True) or a rotating star (False). 

        Returns:
            tuple: tuple containing:

                - **dr** (*float*): The initial stepsize in cm. 
                - **intial**   (*array*): A np array storing the initial conditions.

        """
        if hasattr(pcent, '__len__') or hasattr(epscent, '__len__'):
            # pcent and epscent are sometimes scalars and sometimes arrays of length 1,
            # causing ragged arrays which numpy no longer accepts.
            # Therefore, check if:
            # 1. Either of them are arrays
            # 2. If so, that both of them are
            # 3. That they have the same shape
            # 4. And that that shape is (1,).
            # If this is true, convert them to scalars to avoid ragged arrays.
            try:
                assert(hasattr(pcent, '__len__') and hasattr(epscent, '__len__'))
                assert(pcent.shape == epscent.shape)
                assert(pcent.shape == (1,))
            except AssertionError:
                raise ValueError('The python TOV solver has tried to create a ragged numpy array. This is no longer supported.')
            pcent = pcent[0]
            epscent = epscent[0]

        r = 4.441e-16
        dr = 10.

        P0 = pcent - (2. * jnp.pi / 3.) * (pcent + epscent) * \
            (3. * pcent + epscent) * r**2.
        m0 = 4. / 3. * jnp.pi * epscent * r**3.
        h0 = r**2.
        b0 = 2. * r

        initial = jnp.array([P0, m0, h0, b0, 0.0])

        return dr, initial

@jax.jit
def tidal_deformability(y2, Mns, Rns):

    C = Mns / Rns
    Eps = 4. * C**3. * (13. - 11. * y2 + C * (3. * y2 - 2.) +
                        2. * C**2. * (1. + y2)) + \
        3. * (1. - 2. * C)**2. * (2. - y2 + 2. * C * (y2 - 1.)) * \
        jnp.log(1. - 2. * C) + 2. * C * (6. - 3. * y2 + 3. * C * (5. * y2 - 8.))
    tidal_def = 16. / (15. * Eps) * (1. - 2. * C)**2. *\
        (2. + 2. * C * (y2 - 1.) - y2)

    return tidal_def


def solveTOVde(epscent, rho_plus, alpha, eos_eps, eos_pres, eos_epsde, eos_presde, atol, rtol, hmax, step): #assumed to be in cgs units as inputs eps has units of g/cm^3 and pres has units g/(cm s^2)

    eos_pres, indices = jnp.unique(jnp.log10(eos_pres).round(decimals=3), return_index=True)
    eos_pres = 10**eos_pres * G * jnp.power(c,-4) #scaled into geometrized
    eos_eps = eos_eps[jnp.sort(indices)] * G * jnp.power(c,-2) #scaled into geometrized

    eos_presde, indices = jnp.unique(jnp.log10(eos_presde).round(decimals=5), return_index=True)
    eos_presde = 10**eos_presde * G * jnp.power(c,-4) #scaled into geometrized
    eos_epsde = eos_epsde[jnp.sort(indices)] * G * jnp.power(c,-2) #scaled into geometrized

    # Set initial conditions for solving the TOV equations ##
    epscent = epscent * G * jnp.power(c,-2)
    pcent = epsilon_pressure(epscent, eos_epsde, eos_presde)
    adindcent = pressure_adind(pcent, eos_epsde, eos_presde)

    rho_plus = rho_plus * G * jnp.power(c,-2)
    rho_minus = alpha * rho_plus

    p_plus = epsilon_pressure(rho_plus, eos_epsde, eos_presde)

    r = 4.441e-16
    rmax = 50 * 1e5
    dr, initial = initial_conditions(epscent, pcent, adindcent)

    def stop(r,y,args, **kwargs):
        epsgrid, presgrid = args
        return y[0] - p_plus
    #stop.terminal = True
    #stop.direction = -1.0

    # Integrate the TOV equations

    term = ODETerm(TOV)
    solver = Tsit5()
    t0 = r
    t1 = rmax
    dt0 = None
    y0 = initial
    args = (eos_epsde, eos_presde)
    event = diffrax.Event(stop)
    max_steps = 5000
    max_step_size = 10000.
    stepsize_controller = PIDController(rtol = rtol, atol = atol, dtmax = max_step_size)
    #saveat = SaveAt(t0 = True, t1 = True, steps = True)
    saveat = SaveAt(t1 = True)

    sol_in = diffeqsolve(term, solver, t0, t1, dt0, y0, args=(eos_epsde, eos_presde), saveat=saveat, event=event,
    stepsize_controller=stepsize_controller, max_steps=max_steps )
    #sol_in = solve_ivp(TOV, t_span=(r, rmax), y0=initial,
    #                   method='RK45', t_eval=None, args=(eos_epsde, eos_presde), events = stop,
    #                   max_step=10000.)


    Mde = sol_in.ys[1][0] # in geometrized units
    Rde_core = sol_in.ts[0] #in units of cm

    def stop2(r,y,args, **kwargs):
        epsgrid,presgird,rho_minus = args
        return y[0]
    #stop.terminal = True
    #stop.direction = -1.0

    term = ODETerm(TOV_eps)
    solver = Tsit5()
    t0 = Rde_core
    t1 = rmax
    dt0 = None
    y0 = jnp.array([sol_in.ys[0][0],Mde, sol_in.ys[2][0], sol_in.ys[3][0], sol_in.ys[4][0], rho_minus])
    args = (eos_eps, eos_pres, rho_minus)
    event = diffrax.Event(stop2)
    max_steps = 5000
    max_step_size = 10000.
    stepsize_controller = PIDController(rtol = rtol, atol = atol, dtmax = max_step_size)
    #saveat = SaveAt(t0 = True, t1 = True, steps = True)
    saveat = SaveAt(t1 = True)

    sol_shell = diffeqsolve(term, solver, t0, t1, dt0, y0, args=(eos_eps, eos_pres, rho_minus), saveat=saveat, event=event,
    stepsize_controller=stepsize_controller, max_steps=max_steps )
    

    Mb = sol_shell.ys[1][0]
    Rns =  sol_shell.ts[0]
    y = Rns * sol_shell.ys[3][0] / sol_shell.ys[2][0]
    
    tidal = tidal_deformability(y, Mb, Rns)

    #Gtt is wrong, but not going to use so can leave for now
    Gtt = jnp.zeros((len(sol_in.t), 2))
    #Gtt[:,0] = sol_ins.t #radius in cm
    #Gtt[:,1] = sol_in.ys[4][0] - (sol_in.ys[4][0] - 0.5 * jnp.log(1 - 2 * Mb / Rns))

    Mb = Mb * jnp.power(c,2) / G #now in units of grams
    Mde = Mde * jnp.power(c,2) / G #now in units of grams

    Mb = Mb - Mde

    return Mb, Mde, Rde_core, Rns, tidal, Gtt
