#pragma once
#include <casadi/casadi.hpp>
#include <ocp/StageOCP.hpp>
#include <vector>
#include <map>
#include <unordered_set>
#include "casadi/core/function_internal.hpp"
#include "casadi/core/function.hpp"
#include "casadi/core/code_generator.hpp"
#include "casadi/core/importer.hpp"
#include <limits>
#include "auxiliary.hpp"
#define JIT_HACKED_CASADI 0
namespace fatrop_casadi
{
    typedef int (*eval_t)(const double **arg, double **res,
                          long long int *iw, double *w, int);
    enum stage
    {
        initial,
        path,
        terminal
    };
    template <typename der>
    struct stagequantity
    {
        bool initial = false;
        bool path = false;
        bool terminal = false;
        const der at_initial()
        {
            initial = true;
            return static_cast<der &>(*this);
        }
        const der at_path()
        {
            path = true;
            return static_cast<der &>(*this);
        }
        const der at_terminal()
        {
            terminal = true;
            return static_cast<der &>(*this);
        }
    };

    struct constraint : public stagequantity<constraint>
    {
        casadi::DM lb;
        casadi::MX c;
        casadi::DM ub;
        constraint(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub)
            : lb(lb), c(c), ub(ub)
        {
        }
        static constraint equality(const casadi::MX &c)
        {
            return constraint(casadi::DM::zeros(c.size1(), c.size2()), c, casadi::DM::zeros(c.size1(), c.size2()));
        }
        static constraint lower_bounded(const casadi::DM &lb, const casadi::MX &c)
        {
            return constraint{lb, c, std::numeric_limits<double>::infinity() * casadi::DM::ones(c.size1(), c.size2())};
        }
        static constraint upper_bounded(const casadi::MX &c, const casadi::DM &ub)
        {
            return constraint{-std::numeric_limits<double>::infinity() * casadi::DM::ones(c.size1(), c.size2()), c, ub};
        }
        static constraint box(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub)
        {
            return constraint{lb, c, ub};
        }
    };

    class ConstraintsAuxiliary
    {
    public:
        void seperate_constraints(const casadi::DM &lbs, const casadi::MX &eqs, const casadi::DM &ubs, casadi::MX &eq, casadi::DM &lb, casadi::MX &ineq, casadi::DM &ub)
        {
            using casadi::MX;
            eq = casadi::MX::zeros(0, 0);
            ineq = casadi::MX::zeros(0, 0);
            lb = casadi::DM::zeros(0, 0);
            ub = casadi::DM::zeros(0, 0);
            for (int i = 0; i < eqs.rows(); i++)
            {
                if ((double)lbs(i, 0) == (double)ubs(i, 0))
                {
                    eq = vertcat(eq, eqs(i, 0) - lbs(i, 0));
                }
                else
                {
                    ineq = vertcat(ineq, eqs(i, 0));
                    lb = vertcat(lb, lbs(i, 0));
                    ub = vertcat(ub, ubs(i, 0));
                }
            }
        }
    };

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
    struct SingleStageInternal
    {
        constraint at_t0(const constraint &c)
        {
            constraint res = c;
            return res.at_initial();
        }
        constraint at_tf(const constraint &c)
        {
            constraint res = c;
            return res.at_terminal();
        }
        SingleStageInternal(const int K = 50)
        {
            dims.K = K;
            dirty = true;
        }
        // problem dimensions
        SingleStageDims dims;
        // discrete dynamics function
        casadi::MX state(const std::string &name, int m, int n)
        {
            dirty = true;
            auto x = casadi::MX::sym(name, m, n);
            vec_x.push_back(x);
            map_x_next[x] = casadi::MX::zeros(m, n);
            map_x_initial[x] = casadi::DM::zeros(m * n, 1);
            return x;
        }
        bool depends_on(const casadi::MX &x)
        {
            auto x_vec = veccat(vec_x);
            auto u_vec = veccat(vec_u);
            return casadi::MX::depends_on(x, x_vec) || casadi::MX::depends_on(x, u_vec);
        }
        casadi::MX control(const std::string &name, int m, int n)
        {
            dirty = true;
            auto u = casadi::MX::sym(name, m, n);
            vec_u.push_back(u);
            map_u_initial[u] = casadi::DM::zeros(m * n, 1);
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
        void subject_to(const constraint &constraints)
        {
            std::unordered_set<stage> stages;
            if (constraints.initial)
                stages.insert(stage::initial);
            if (constraints.path)
                stages.insert(stage::path);
            if (constraints.terminal)
                stages.insert(stage::terminal);
            subject_to(constraints.lb, constraints.c, constraints.ub, stages);
        }
        void subject_to(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub, const std::unordered_set<stage> &stages)
        {
            bool initial = stages.find(stage::initial) != stages.end();
            bool middle = stages.find(stage::path) != stages.end();
            bool terminal = stages.find(stage::terminal) != stages.end();
            subject_to(lb, c, ub, initial, middle, terminal);
        }
        void subject_to(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub, bool initial, bool middle, bool terminal)
        {
            dirty = true;
            if (initial)
            {
                stage_properties_initial.vec_c.push_back(c);
                stage_properties_initial.lb_c.push_back(lb);
                stage_properties_initial.ub_c.push_back(ub);
            }
            if (middle)
            {
                stage_properties_path.vec_c.push_back(c);
                stage_properties_path.lb_c.push_back(lb);
                stage_properties_path.ub_c.push_back(ub);
            }
            if (terminal)
            {
                stage_properties_terminal.vec_c.push_back(c);
                stage_properties_terminal.lb_c.push_back(lb);
                stage_properties_terminal.ub_c.push_back(ub);
            }
        }
        void add_objective(const casadi::MX &c, const std::unordered_set<stage> &stages)
        {
            bool initial = stages.find(stage::initial) != stages.end();
            bool middle = stages.find(stage::path) != stages.end();
            bool terminal = stages.find(stage::terminal) != stages.end();
            add_objective(c, initial, middle, terminal);
        }
        void add_objective(const casadi::MX &c, bool initial, bool middle, bool terminal)
        {
            dirty = true;
            if (initial)
            {
                stage_properties_initial.obj.push_back(c);
            }
            if (middle)
            {
                stage_properties_path.obj.push_back(c);
            }
            if (terminal)
            {
                stage_properties_terminal.obj.push_back(c);
            }
        };
        void set_next(const casadi::MX &x, const casadi::MX &x_next)
        {
            dirty = true;
            map_x_next[x] = x_next;
        }
        void set_initial(const casadi::MX &x, const casadi::DM &value)
        {
            // check if in state map
            if (map_x_initial.find(x) != map_x_initial.end())
                map_x_initial[x] = value;
            else if (map_u_initial.find(x) != map_u_initial.end())
                map_u_initial[x] = value;
            else
                throw std::runtime_error("set_initial: variable not found");
        }
        void set_value(const casadi::MX &x, const casadi::DM &value)
        {
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
        std::map<casadi::MX, casadi::MX, comp_mx> map_x_next;
        std::map<casadi::MX, casadi::DM, comp_mx> map_x_initial;
        std::map<casadi::MX, casadi::DM, comp_mx> map_u_initial;

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
        std::vector<double> initial_x;
        std::vector<double> initial_u;
        casadi::MX sum(const std::vector<casadi::MX> &vec)
        {
            casadi::MX sum = casadi::MX::zeros(1, 1);
            for (auto &x : vec)
                sum += x;
            return sum;
        }
        void make_clean()
        {
            using casadi::MX;
            auto x = veccat(vec_x);
            auto u = veccat(vec_u);
            auto p = veccat(vec_p);
            auto p_stage = veccat(vec_p_stage);
            dims.nx = (int)x.numel();
            dims.nu = (int)u.numel();
            dims.n_global_params = (int)p.numel();
            dims.n_stage_params = (int)p_stage.numel();

            std::vector<casadi::MX> vec_x_next;
            for (auto &x : map_x_next)
            {
                vec_x_next.push_back(x.second);
            }
            auto x_next = veccat(vec_x_next);
            // make the dynamics function
            dynamics_func = casadi::Function("dynamics", {x, u, p, p_stage}, {x_next}, {"x", "u", "p", "p_stage"}, {"x_next"});
            std::vector<stage_functions *> stage_fcs = {&stage_functions_initial, &stage_functions_path, &stage_functions_terminal};
            std::vector<stage_properties *> stage_props = {&stage_properties_initial, &stage_properties_path, &stage_properties_terminal};
            std::vector<int *> ng_v = {&dims.ngI, &dims.ng, &dims.ngF};
            std::vector<int *> ngineq_v = {&dims.ng_ineqI, &dims.ng_ineq, &dims.ng_ineqF};
            // initial stage
            for (int i = 0; i < 3; i++)
            {
                auto &stage_functions_curr = *stage_fcs.at(i);
                auto &stage_properties_curr = *stage_props.at(i);
                stage_functions_curr.cost = casadi::Function("cost", {x, u, p, p_stage}, {sum(stage_properties_curr.obj)}, {"x", "u", "p", "p_stage"}, {"cost"});
                auto eqs = casadi::MX::veccat(stage_properties_curr.vec_c);
                auto lbs = casadi::DM::veccat(stage_properties_curr.lb_c);
                auto ubs = casadi::DM::veccat(stage_properties_curr.ub_c);
                auto eq = casadi::MX::zeros(0, 1);
                casadi::DM lb;
                casadi::DM ub;
                auto ineq = casadi::MX::zeros(0, 1);
                ConstraintsAuxiliary().seperate_constraints(lbs, eqs, ubs, eq, lb, ineq, ub);
                stage_functions_curr.eq = casadi::Function("eq", {x, u, p, p_stage}, {eq}, {"x", "u", "p", "p_stage"}, {"eq"});
                stage_functions_curr.ineq = casadi::Function("ineq", {x, u, p, p_stage}, {ineq}, {"x", "u", "p", "p_stage"}, {"ineq"});
                stage_functions_curr.ineq_lb = lb;
                stage_functions_curr.ineq_ub = ub;
                *ng_v.at(i) = (int)eq.numel();
                *ngineq_v.at(i) = (int)ineq.numel();
            }
        }
        // add all variables of map_initial_x to initial_x
        bool dirty = true;
    };
    struct SingleStage : public SharedObj<SingleStageInternal>
    {
        using SharedObj<SingleStageInternal>::SharedObj;
        constraint at_t0(const constraint &c)
        {
            return get()->at_t0(c);
        }
        constraint at_tf(const constraint &c)
        {
            return get()->at_tf(c);
        }
        casadi::MX state(const std::string &name, int m, int n)
        {
            return get()->state(name, m, n);
        }
        bool depends_on(const casadi::MX &x)
        {
            return get()->depends_on(x);
        }
        casadi::MX control(const std::string &name, int m, int n)
        {
            return get()->control(name, m, n);
        }
        casadi::MX parameter(const std::string &name, int m, int n)
        {
            return get()->parameter(name, m, n);
        }
        casadi::MX stage_parameter(const std::string &name, int m, int n)
        {
            return get()->stage_parameter(name, m, n);
        }
        void subject_to(const constraint &constraints)
        {
            get()->subject_to(constraints);
        }
        void subject_to(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub, const std::unordered_set<stage> &stages)
        {
            get()->subject_to(lb, c, ub, stages);
        }
        void subject_to(const casadi::DM &lb, const casadi::MX &c, const casadi::DM &ub, bool initial, bool middle, bool terminal)
        {
            get()->subject_to(lb, c, ub, initial, middle, terminal);
        }
        void add_objective(const casadi::MX &c, const std::unordered_set<stage> &stages)
        {
            get()->add_objective(c, stages);
        }
        void add_objective(const casadi::MX &c, bool initial, bool middle, bool terminal)
        {
            get()->add_objective(c, initial, middle, terminal);
        };
        void set_next(const casadi::MX &x, const casadi::MX &x_next)
        {
            get()->set_next(x, x_next);
        }
        void set_initial(const casadi::MX &x, const casadi::DM &value)
        {
            get()->set_initial(x, value);
        }
        void set_value(const casadi::MX &x, const casadi::DM &value)
        {
            this->get()->set_value(x, value);
        }
        void make_clean()
        {
            this->get()->make_clean();
        }
    };


} // namespace fatrop_casadi