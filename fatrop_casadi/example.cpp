#include "single_stage.hpp"
#include <casadi/casadi.hpp>
#include <ocp/StageOCPApplication.hpp>
#include <memory>

using namespace fatrop_casadi;
int main()
{
    auto ocp = SingleStage(50);
    auto x1 = ocp.state("x1", 1, 1);
    auto x2 = ocp.state("x2", 1, 1);
    auto u = ocp.control("u", 1, 1);
    auto e = 1 - x1 * x1 - x2 * x2;
    double dt = 0.02;
    ocp.set_next(x1, (e*x1 - x2 + u)*dt);
    ocp.set_next(x2, x1*dt);
    ocp.add_objective(u*u, true, true, true);
    ocp.subject_to({0}, x1 - 1, {0}, true, false, false);
    ocp.subject_to({0}, x2 - 2, {0}, true, false, false);



    auto opti = SingleStageOptiAdapter(ocp).opti;
    opti.solver("ipopt");
    opti.solve();

    ocp.make_clean();
    auto fatrop = std::make_shared<SingleStageFatropAdapter>(ocp);
    auto app = fatrop::StageOCPApplication(fatrop);
    app.build();
    app.optimize();

    return 0;
}