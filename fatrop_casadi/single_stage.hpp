#pragma once
#include <casadi/casadi.hpp>
#include <ocp/OCPAbstract.hpp>
#include <vector>

namespace fatrop_casadi
{
    struct SingleStageDims
    {
        int nu;
        int nx;
        int ngI;
        int ng;
        int ngF;
        int ng_ineqI;
        int ng_ineq;
        int ng_ineqF;
        int n_stage_params;
        int n_global_params;
        int K;
    };
    struct SingleStage
    {
        // problem dimensions
        const SingleStageDims dims;
        // discrete dynamics function
        const casadi::Function dynamics;
        // cost function
        const casadi::Function costI;
        const casadi::Function costP;
        const casadi::Function costF;
        // inequality constraints
        const casadi::Function ineqI;
        const std::vector<double> ineqI_lb;
        const std::vector<double> ineqI_ub;
        const casadi::Function ineqP;
        const std::vector<double> ineqP_lb;
        const std::vector<double> ineqP_ub;
        const casadi::Function ineqF;
        const std::vector<double> ineqF_lb;
        const std::vector<double> ineqF_ub;
        // equality constraints`
        const casadi::Function eqI;
        const casadi::Function eqP;
        const casadi::Function eqF;
        // initial guess
        // default parameters
    };
    class SingleStageOptiAdapter
    {
        SingleStageOptiAdapter(const SingleStage &ss)
        {
            // add x variables to opti
            for (int k = 0; k < ss.dims.K; k++)
                x.push_back(opti.variable(ss.dims.nx));
            // add u variables to opti
            for (int k = 0; k < ss.dims.K - 1; k++)
                u.push_back(opti.variable(ss.dims.nu));
            // add the stage parameters to opti
            for (int k = 0; k < ss.dims.K; k++)
                stage_params.push_back(opti.parameter(ss.dims.n_stage_params));
            // add the global parameters to opti
            global_params = opti.parameter(ss.dims.n_global_params);

            // add the dynamics constraints
            for (int k = 0; k < ss.dims.K - 1; k++)
                opti.subject_to(casadi::Opti::bounded(0, x[k + 1] - ss.dynamics({x[k], u[k], stage_params[k], global_params})[0], 0));
            // add the initial eq constraint
            opti.subject_to(casadi::Opti::bounded(0, ss.eqI({x[0], u[0], stage_params[0], global_params})[0], 0));
            // add the path eq constraints
            for (int k = 1; k < ss.dims.K - 1; k++)
                opti.subject_to(casadi::Opti::bounded(0, ss.eqP({x[k], u[k], stage_params[k], global_params})[0], 0));
            // add the final eq constraint
            opti.subject_to(casadi::Opti::bounded(0, ss.eqF({x[ss.dims.K - 1], stage_params[ss.dims.K - 1], global_params})[0], 0));

        };
        casadi::Opti opti;
        std::vector<casadi::MX> x;
        std::vector<casadi::MX> u;
        std::vector<casadi::MX> stage_params;
        casadi::MX global_params;
    };
    class SingleStageFatropAdapter : public fatrop::OCPAbstract
    {
    };
} // namespace fatrop_casadi