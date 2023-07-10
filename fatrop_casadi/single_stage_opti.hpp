
#pragma once
#include <casadi/casadi.hpp>
#include <vector>
#include <map>
#include <unordered_set>
#include <limits>
#include "single_stage.hpp"
#define JIT_HACKED_CASADI 0
namespace fatrop_casadi
{
    class SingleStageOptiAdapter
    {
    public:
        SingleStageOptiAdapter(SingleStage &ss)
        {
            ss.make_clean();
            // add variables
            add_variables(ss, opti);
            // add constraints
            add_constraints(ss, opti);
            // add objective
            add_objective(ss, opti);
        };
        void add_variables(const SingleStage &ss, casadi::Opti &opti)
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
        }
        void add_constraints(const SingleStage &ss, casadi::Opti &opti)
        {

            // add the dynamics constraints
            for (int k = 0; k < ss.dims.K - 1; k++)
                opti.subject_to(casadi::Opti::bounded(0, x[k + 1] - ss.dynamics_func({x[k], u[k], stage_params[k], global_params})[0], 0));
            // add the initial eq constraint
            if (ss.dims.ngI > 0)
                opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_initial.eq({x[0], u[0], stage_params[0], global_params})[0], 0));
            if (ss.dims.ng_ineqI > 0)
                opti.subject_to(casadi::Opti::bounded(ss.stage_functions_initial.ineq_lb, ss.stage_functions_initial.ineq({x[0], u[0], stage_params[0], global_params})[0], ss.stage_functions_initial.ineq_ub));
            // add the path eq constraints
            for (int k = 1; k < ss.dims.K - 1; k++)
            {
                if (ss.dims.ng > 0)
                    opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_path.eq({x[k], u[k], stage_params[k], global_params})[0], 0));
                if (ss.dims.ng_ineq > 0)
                    opti.subject_to(casadi::Opti::bounded(ss.stage_functions_path.ineq_lb, ss.stage_functions_path.ineq({x[k], u[k], stage_params[k], global_params})[0], ss.stage_functions_path.ineq_ub));
            }
            // add the final eq constraint
            if (ss.dims.ngF > 0)
                opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_terminal.eq({x[0], u[0], stage_params[0], global_params})[0], 0));
            if (ss.dims.ng_ineqF > 0)
                opti.subject_to(casadi::Opti::bounded(ss.stage_functions_terminal.ineq_lb, ss.stage_functions_terminal.ineq({x[ss.dims.K - 1], u[ss.dims.K - 2], stage_params[ss.dims.K - 1], global_params})[0], ss.stage_functions_terminal.ineq_ub));
        }
        void add_objective(const SingleStage &ss, casadi::Opti &opti)
        {
            auto J = casadi::MX::zeros(1);
            // add the initial objective
            J += ss.stage_functions_initial.cost({x[0], u[0], stage_params[0], global_params})[0];
            // add the path objective
            for (int k = 1; k < ss.dims.K - 1; k++)
                J += ss.stage_functions_path.cost({x[k], u[k], stage_params[k], global_params})[0];
            // add the final objective
            J += ss.stage_functions_terminal.cost({x[ss.dims.K - 1], u[ss.dims.K - 2], stage_params[ss.dims.K - 1], global_params})[0];
            opti.minimize(J);
        }
        casadi::Opti opti;
        std::vector<casadi::MX> x;
        std::vector<casadi::MX> u;
        std::vector<casadi::MX> stage_params;
        casadi::MX global_params;
    };
}