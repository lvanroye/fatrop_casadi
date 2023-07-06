import casadi as cs
from rockit import *

# auto ocp = SingleStage(50);
# auto x1 = ocp.state("x1", 1, 1);
# auto x2 = ocp.state("x2", 1, 1);
# auto u = ocp.control("u", 1, 1);
# auto e = 1 - x1 * x1 - x2 * x2;
# double dt = 0.02;
# ocp.set_next(x1, (e*x1 - x2 + u)*dt);
# ocp.set_next(x2, x1*dt);
# ocp.add_objective(u*u, true, true, true);
# ocp.subject_to({0}, x1 - 1, {0}, true, false, false);
# ocp.subject_to({0}, x2 - 2, {0}, true, false, false);
# ocp.subject_to({-1000}, x2, {2}, false, true, true);

dt = 0.02
ocp = Ocp(T=50*dt)
x1 = ocp.state()
x2 = ocp.state()
u = ocp.control()
e = 1 - x1 * x1 - x2 * x2
ocp.set_next(x1, (e * x1 - x2 + u) * dt)
ocp.set_next(x2, x1 * dt)
ocp.add_objective(ocp.sum(u * u, True))
ocp.subject_to(ocp.at_t0(x1) == 1)
ocp.subject_to(ocp.at_t0(x2) == 2)
ocp.subject_to( -1000 <=(x2 <= 2), include_first=False)

ocp.method(external_method("fatrop",N = 50))
ocp.solve()
ocp._method.myOCP.set_option("mu_init", 1e-1)
ocp.solve()
