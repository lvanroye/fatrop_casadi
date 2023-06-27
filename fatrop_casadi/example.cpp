#include "single_stage.hpp"
#include <casadi/casadi.hpp>

int main()
{
    // create a simple test opti problem
    casadi::Opti opti;
    // add x variables to opti
    auto x =  opti.variable(1);
    // set objective
    opti.minimize(x*x);
    // create solver instance
    opti.solver("ipopt");
    // solve the problem
    auto sol = opti.solve();
    sol = opti.solve();
    return 0;
}