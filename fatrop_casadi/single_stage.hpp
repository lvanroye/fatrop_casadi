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
    class MXWrap : public casadi::MX
    {
    public:
        MXWrap(const casadi::MX &mx, const int id) : casadi::MX(mx), id(id){};
        const int id;
    };
    struct SingleStage
    {
        // problem dimensions
        SingleStageDims dims;
        // discrete dynamics function
        MXWrap state(const std::string &name, int m, int n)
        {
            dirty = true;
            auto x = casadi::MX::sym(name, m, n);
            vec_x.push_back(x);
            vec_x_next.push_back(casadi::MX::zeros(m, n));
            return MXWrap(x, vec_x.size() - 1);
        }
        casadi::MX control(const std::string &name, int m, int n)
        {
            dirty = true;
            auto u = casadi::MX::sym(name, m, n);
            vec_u.push_back(u);
            return u;
        }
        casadi::MX parameter(const std::string &name, int m, int n)
        {
            dirty = true;
            auto p = casadi::MX::sym(name, m, n);
            vec_p.push_back(p);
            return p;
        }
        casadi::MX stage_parameter(const std::string &name, int m, int n)
        {
            dirty = true;
            auto p = casadi::MX::sym(name, m, n);
            vec_p_stage.push_back(p);
            return p;
        }
        void subject_to(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub, bool initial, bool middle, bool terminal)
        {
            dirty = true;
        }
        void add_objective(const casadi::MX &c, bool initial, bool middle, bool terminal)
        {
            dirty = true;
        };
        void set_next(const MXWrap &x, const casadi::MX &x_next)
        {
            dirty = true;
            vec_x_next[x.id] = x_next;
        }
        std::vector<casadi::MX> vec_x;
        std::vector<casadi::MX> vec_u;
        std::vector<casadi::MX> vec_p;
        std::vector<casadi::MX> vec_p_stage;
        struct stage_properties
        {
            std::vector<casadi::MX> vec_c;
            std::vector<casadi::DM> lb_c;
            std::vector<casadi::DM> ub_c;
            std::vector<casadi::MX> obj_initial;
        };
        std::vector<casadi::MX> vec_x_next;
        stage_properties stage_properties_initial;
        stage_properties stage_properties_path;
        stage_properties stage_properties_terminal;
        const casadi::Function dynamics_func;
        struct stage_functions
        {
            const casadi::Function cost;
            const casadi::Function ineq;
            const std::vector<double> ineq_lb;
            const std::vector<double> ineq_ub;
            const casadi::Function eq;
        };
        stage_functions stage_functions_initial;
        stage_functions stage_functions_path;
        stage_functions stage_functions_terminal;
        bool dirty = true;
    };
    class SingleStageOptiAdapter
    {
    public:
        SingleStageOptiAdapter(const SingleStage &ss)
        {
            // add variables
            add_variables(ss, opti);
            // add constraints
            add_constraints(ss, opti);
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
            opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_initial.eq({x[0], u[0], stage_params[0], global_params})[0], 0));
            opti.subject_to(casadi::Opti::bounded(ss.stage_functions_initial.ineq_lb, ss.stage_functions_initial.ineq({x[0], u[0], stage_params[0], global_params})[0], ss.stage_functions_initial.ineq_ub));
            // add the path eq constraints
            for (int k = 1; k < ss.dims.K - 1; k++)
            {
                opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_path.eq({x[k], u[k], stage_params[k], global_params})[0], 0));
                opti.subject_to(casadi::Opti::bounded(ss.stage_functions_path.ineq_lb, ss.stage_functions_path.ineq({x[k], u[k], stage_params[k], global_params})[0], ss.stage_functions_path.ineq_ub));
            }
            // add the final eq constraint
            opti.subject_to(casadi::Opti::bounded(0, ss.stage_functions_terminal.eq({x[0], u[0], stage_params[0], global_params})[0], 0));
            opti.subject_to(casadi::Opti::bounded(ss.stage_functions_terminal.ineq_lb, ss.stage_functions_terminal.ineq({x[ss.dims.K-1], u[ss.dims.K-2], stage_params[ss.dims.K-1], global_params})[0], ss.stage_functions_terminal.ineq_ub));
        }
        casadi::Opti opti;
        std::vector<casadi::MX> x;
        std::vector<casadi::MX> u;
        std::vector<casadi::MX> stage_params;
        casadi::MX global_params;
    };
    // class SingleStageFatropAdapter : public fatrop::StageOCP
    // {
    // };
} // namespace fatrop_casadi