#include "single_stage.hpp"
#include <casadi/casadi.hpp>
#include <ocp/StageOCPApplication.hpp>
#include <memory>
#include <limits>
#define INF std::numeric_limits<double>::infinity()

using namespace fatrop_casadi;
int main()
{
    auto ocp = SingleStage(50);
    auto x1 = ocp.state("x1", 1, 1);
    auto x2 = ocp.state("x2", 1, 1);
    auto u = ocp.control("u", 1, 1);
    auto e = 1 - x1 * x1 - x2 * x2;
    double dt = 0.02;
    ocp.set_next(x1, (e * x1 - x2 + u) * dt);
    ocp.set_next(x2, x1 * dt);
    ocp.add_objective(u * u, {stage::initial, stage::path, stage::terminal});
    ocp.subject_to(constraint::equality(x1 - 1).at_initial());
    ocp.subject_to(constraint::equality(x2 - 2).at_initial());
    ocp.subject_to(constraint::upper_bounded(x1, 1).at_path());
    ocp.subject_to(constraint::upper_bounded(x2, 2).at_terminal());

    ocp.make_clean();
    auto fatrop = std::make_shared<SingleStageFatropAdapter>(ocp, casadi::Dict());
    auto app = fatrop::StageOCPApplication(fatrop);
    app.build();
    app.set_option("mu_init", 1e-1);
    app.optimize();
    return 0;
}