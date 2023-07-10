
#include "single_stage.hpp"
#include "single_stage_opti.hpp"
#include "single_stage_fatrop.hpp"
#include "rk4.hpp"
#include <casadi/casadi.hpp>
#include <ocp/StageOCPApplication.hpp>
#include <memory>
#include <limits>
casadi::MX transf(const casadi::MX &theta, const casadi::MX &p)
{
    return casadi::MX::vertcat({casadi::MX::horzcat({casadi::MX::cos(theta), -casadi::MX::sin(theta), p(0)}),
                                casadi::MX::horzcat({casadi::MX::sin(theta), casadi::MX::cos(theta), p(1)}),
                                casadi::MX::horzcat({0.0, 0.0, 1.})});
}
using namespace fatrop_casadi;
int main()
{
    auto ocp = SingleStage(50);
    double dt = 0.01;
    double m = 1.0;
    double g = 9.81;
    double I = 0.1;
    double D = 1.0;
    double max_thrust = 2 * g;

    auto p = ocp.state("p", 2, 1);
    auto dp = ocp.state("dp", 2, 1);
    auto theta = ocp.state("theta", 1, 1);
    auto dtheta = ocp.state("dtheta", 1, 1);

    auto F1 = ocp.control("F1", 1, 1);
    auto F2 = ocp.control("F2", 1, 1);

    auto F_r = transf(theta, p);


    auto F_tot = (casadi::MX::mtimes(F_r, casadi::MX::vertcat({0, F1 + F2, 0})))(casadi::Slice(0, 2), 0);
    auto d_p =  dp ;
    auto d_theta =  dtheta;
    auto d_dp =  (1 / m * F_tot + casadi::MX::vertcat({0, -g})) ;
    auto d_dtheta =  (1 / I * D / 2 * (F2 - F1)) ;
    auto integrator = rk4({{p,d_p}, {theta, d_theta}, {dp, d_dp}, {dtheta, d_dtheta}}, dt);

    ocp.set_next(p, integrator(p));
    ocp.set_next(theta, integrator(theta));
    ocp.set_next(dp, integrator(dp));
    ocp.set_next(dtheta, integrator(dtheta));

    // # Define the path constraints
    ocp.subject_to(constraint::box(0, F1, max_thrust).at_initial());
    ocp.subject_to(constraint::box(0, F2, max_thrust).at_initial());
    ocp.subject_to(constraint::box(0, F1, max_thrust).at_path());
    ocp.subject_to(constraint::box(0, F2, max_thrust).at_path());
    ocp.add_objective(1e2*(F1 * F1 + F2 * F2), {stage::initial, stage::path});
    ocp.add_objective(1e0*casadi::MX::sumsqr(p - casadi::DM({5., 5.})), {stage::terminal});

    //     target = np.array([5., 5.])
    // # Define the initial conditions
    ocp.subject_to(ocp.at_t0(constraint::equality(p)));
    ocp.subject_to(ocp.at_t0(constraint::equality(dp)));
    ocp.subject_to(ocp.at_t0(constraint::equality(theta)));
    ocp.subject_to(ocp.at_t0(constraint::equality(dtheta)));

    // # Define the end conditions
    // ocp.subject_to(ocp.at_tf(constraint::equality(p - casadi::DM({1, 1}))));
    // self.ocp.subject_to(self.ocp.at_tf(self.dp) == [0, 0])

    ocp.make_clean();


    auto fatrop = std::make_shared<SingleStageFatropAdapter>(ocp, casadi::Dict());
    auto app = fatrop::StageOCPApplication(fatrop);
    app.build();
    app.set_option("mu_init", 1e-1);
    // app.set_option("max_iter", 1000);
    app.optimize();
    return 0;
}