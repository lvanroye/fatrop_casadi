#include "single_stage.hpp"
#include "multi_stage.hpp"
#include "single_stage_opti.hpp"
#include "single_stage_fatrop.hpp"
#include <casadi/casadi.hpp>
#include <ocp/StageOCPApplication.hpp>
#include <memory>
#include <limits>

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
    ocp.subject_to(ocp.at_t0(constraint::equality(x1 - 1)));
    ocp.subject_to(ocp.at_t0(constraint::equality(x2 - 2)));
    ocp.subject_to(constraint::upper_bounded(x1, 1).at_path());
    ocp.subject_to(ocp.at_tf(constraint::upper_bounded(x2, 2)));
    ocp.make_clean();

    auto fatrop = SingleStageFatropAdapter(ocp, casadi::Dict());
    auto app = fatrop::StageOCPApplication(fatrop);
    app.build();
    app.set_option("mu_init", 1e-1);
    app.optimize();
    return 0;
}