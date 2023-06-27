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
            std::vector<casadi::MX> obj;
        };
        std::vector<casadi::MX> vec_x_next;
        stage_properties stage_properties_initial;
        stage_properties stage_properties_path;
        stage_properties stage_properties_terminal;
        casadi::Function dynamics_func;
        struct stage_functions
        {
            casadi::Function cost;
            casadi::Function ineq;
            casadi::DM ineq_lb;
            casadi::DM ineq_ub;
            casadi::Function eq;
        };
        stage_functions stage_functions_initial;
        stage_functions stage_functions_path;
        stage_functions stage_functions_terminal;
        casadi::MX sum(const std::vector<casadi::MX> &vec)
        {
            casadi::MX sum = casadi::MX::zeros(1, 1);
            for (auto &x : vec)
                sum += x;
            return sum;
        }
        void seperate_constraints(const casadi::DM &lbs, const casadi::MX &eqs, const casadi::DM &ubs, casadi::MX &eq, casadi::DM &lb, casadi::MX &ineq, casadi::DM &ub)
        {
            for (int i = 0; i < eqs.rows(); i++)
            {
                if ((double) lbs(i, 0) == (double) ubs(i, 0))
                {
                    eq = vertcat(eq, eqs(i, 0) - lbs(i, 0));
                }
                else
                {
                    ineq = vertcat(ineq, eqs(i, 0));
                    lb = vertcat(lbs, lbs(i, 0));
                    ub = vertcat(ubs, ubs(i, 0));
                }
            }
        }
        void make_clean()
        {
            using casadi::MX;
            auto x = veccat(vec_x);
            auto u = veccat(vec_u);
            auto p = veccat(vec_p);
            auto p_stage = veccat(vec_p_stage);
            auto x_next = veccat(vec_x_next);
            // make the dynamics function
            dynamics_func = casadi::Function("dynamics", {x, u, p, p_stage}, {x_next}, {"x", "u", "p", "p_stage"}, {"x_next"});
            std::vector<stage_functions*> stage_fcs = {&stage_functions_initial, &stage_functions_path, &stage_functions_terminal}; 
            std::vector<stage_properties*> stage_props = {&stage_properties_initial, &stage_properties_path, &stage_properties_terminal};
            // initial stage
            for(int i=0; i<3; i++)
            {
                auto& stage_functions_curr = *stage_fcs.at(i);
                auto& stage_properties_curr = *stage_props.at(i);
                stage_functions_curr.cost = casadi::Function("cost", {x, u, p, p_stage}, {sum(stage_properties_curr.obj)}, {"x", "u", "p", "p_stage"}, {"cost"});
                auto eqs = casadi::MX::veccat(stage_properties_curr.vec_c);
                auto lbs = casadi::DM::veccat(stage_properties_curr.lb_c);
                auto ubs = casadi::DM::veccat(stage_properties_curr.ub_c);
                auto eq = casadi::MX::zeros(0, 1);
                casadi::DM lb;
                casadi::DM ub;
                auto ineq = casadi::MX::zeros(0, 1);
                seperate_constraints(lbs, eqs, ubs, eq, lb, ineq, ub);
                stage_functions_curr.eq = casadi::Function("eq", {x, u, p, p_stage}, {eq}, {"x", "u", "p", "p_stage"}, {"eq"});
                stage_functions_curr.ineq = casadi::Function("ineq", {x, u, p, p_stage}, {ineq}, {"x", "u", "p", "p_stage"}, {"ineq"});
                stage_functions_curr.ineq_lb = lb;
                stage_functions_curr.ineq_ub = ub;
            }
        }
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
    // class SingleStageFatropAdapter : public fatrop::StageOCP
    // {
    // };
} // namespace fatrop_casadi