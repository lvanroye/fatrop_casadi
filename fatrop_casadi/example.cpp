#include "single_stage.hpp"
#include <casadi/casadi.hpp>

int main()
{
    auto x = casadi::MX::sym("x", 1, 1);
    auto u = casadi::MX::sym("u", 1, 1);
    auto p = x;
    std::cout << u.get() << std::endl;
    std::cout << x.get() << std::endl;
    std::cout << p.get() << std::endl;
    // // create a simple test opti problem
    // casadi::Opti opti;
    // // add x variables to opti
    // auto x =  opti.variable(1);
    // // set objective
    // opti.minimize(x*x);
    // // create solver instance
    // opti.solver("ipopt");
    // // solve the problem
    // auto sol = opti.solve();
    // sol = opti.solve();
    return 0;
}