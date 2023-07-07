#pragma once
#include <casadi/casadi.hpp>
#include <ocp/StageOCP.hpp>
#include <vector>
#include <map>
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
    struct comp_mx
    {
        bool operator()(const casadi::MX &a, const casadi::MX &b) const
        {
            return a.get() < b.get();
        }
    };
    struct SingleStage
    {
        SingleStage(const int K)
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
            return x;
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
                seperate_constraints(lbs, eqs, ubs, eq, lb, ineq, ub);
                stage_functions_curr.eq = casadi::Function("eq", {x, u, p, p_stage}, {eq}, {"x", "u", "p", "p_stage"}, {"eq"});
                stage_functions_curr.ineq = casadi::Function("ineq", {x, u, p, p_stage}, {ineq}, {"x", "u", "p", "p_stage"}, {"ineq"});
                stage_functions_curr.ineq_lb = lb;
                stage_functions_curr.ineq_ub = ub;
                *ng_v.at(i) = (int)eq.numel();
                *ngineq_v.at(i) = (int)ineq.numel();
            }
        }
        bool dirty = true;
    };
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
    class SingleStageFatropAdapter : public fatrop::StageOCP
    {
    public:
        SingleStageFatropAdapter(SingleStage &ss, const casadi::Dict &opts = {}) : fatrop::StageOCP(ss.dims.nu, ss.dims.nx, ss.dims.ngI, ss.dims.ng, ss.dims.ngF, ss.dims.ng_ineqI, ss.dims.ng_ineq, ss.dims.ng_ineqF, ss.dims.n_stage_params, ss.dims.n_global_params, ss.dims.K), arg(7)
        {
            auto x_sym = casadi::MX::sym("x", ss.dims.nx);
            auto xp1_sym = casadi::MX::sym("xp1", ss.dims.nx);
            auto u_sym = casadi::MX::sym("u", ss.dims.nu);
            auto stage_params_sym = casadi::MX::sym("stage_params", ss.dims.n_stage_params);
            auto global_params_sym = casadi::MX::sym("global_params", ss.dims.n_global_params);
            auto BAbt = casadi::MX::zeros(ss.dims.nu + ss.dims.nx + 1, ss.dims.nx);
            auto b = casadi::MX::zeros(ss.dims.nx, 1);
            auto x_next_sym = ss.dynamics_func({x_sym, u_sym, stage_params_sym, global_params_sym})[0];
            b = -xp1_sym + x_next_sym;
            BAbt(casadi::Slice(0, ss.dims.nu + ss.dims.nx), casadi::Slice(0, ss.dims.nx)) = casadi::MX::jacobian(ss.dynamics_func({x_sym, u_sym, stage_params_sym, global_params_sym})[0], casadi::MX::vertcat({u_sym, x_sym})).T();
            BAbt(ss.dims.nu + ss.dims.nx, casadi::Slice(0, ss.dims.nx)) = b;
            eval_BAbtk_func = eval_bf(casadi::Function("eval_BAbtk", {x_sym, xp1_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(BAbt)}, opts));
            eval_bk_func = eval_bf(casadi::Function("eval_bk", {x_sym, xp1_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(b)}, opts));
            std::vector<stageproperties *> sp_vec = {&sp_initial, &sp_middle, &sp_terminal};
            std::vector<SingleStage::stage_functions *> sf_vec = {&ss.stage_functions_initial, &ss.stage_functions_path, &ss.stage_functions_terminal};
            std::vector<int> ng_vec = {ss.dims.ngI, ss.dims.ng, ss.dims.ngF};
            std::vector<int> ng_ineq_vec = {ss.dims.ng_ineqI, ss.dims.ng_ineq, ss.dims.ng_ineqF};
            for (int i = 0; i < 3; i++)
            {
                auto sp_p = sp_vec[i];
                auto sf_p = sf_vec[i];
                const int nu = (i == 2) ? 0 : ss.dims.nu;
                const int ng = ng_vec[i];
                const int ng_ineq = ng_ineq_vec[i];
                auto ux = (i == 2) ? x_sym : casadi::MX::vertcat({u_sym, x_sym});
                // compute RSQrqt
                auto obj = sf_p->cost({x_sym, u_sym, stage_params_sym, global_params_sym})[0];
                sp_p->eval_Lk_func = eval_bf(casadi::Function("eval_Lk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(obj)}, opts));
                casadi::MX rq;
                auto RSQ = casadi::MX::hessian(obj, ux, rq);
                sp_p->eval_rqk_func = eval_bf(casadi::Function("eval_rqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(rq)}, opts));
                // lagrangian contribution dynamics constraints
                auto dual_dyn_sym = casadi::MX::sym("dual_dyn", ss.dims.nx);
                if (i != 2)
                {
                    auto rq_dyn = casadi::MX::zeros(nu + ss.dims.nx, 1);
                    auto RSQlag = casadi::MX::hessian(casadi::MX::dot(dual_dyn_sym, x_next_sym), ux, rq_dyn);
                    RSQ += RSQlag;
                    rq += rq_dyn;
                }
                // lagrangian contribution of euqality constraints
                auto dual_eq_sym = casadi::MX::sym("dual_eq", ng);
                if (ng > 0)
                {
                    auto eq_sym = sf_p->eq({x_sym, u_sym, stage_params_sym, global_params_sym});
                    auto eq = eq_sym[0];
                    auto rq_eq = casadi::MX::zeros(nu + ss.dims.nx, 1);
                    auto RSQlag = casadi::MX::hessian(casadi::MX::dot(dual_eq_sym, eq), ux, rq_eq);
                    RSQ += RSQlag;
                    rq += rq_eq;
                    sp_p->eval_gk_func = eval_bf(casadi::Function("eval_eqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(eq)}, opts));
                    auto Ggt = casadi::MX::zeros(nu + ss.dims.nx + 1, ng);
                    Ggt(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, ng)) = casadi::MX::jacobian(eq, ux).T();
                    Ggt(nu + ss.dims.nx, casadi::Slice(0, ng)) = eq.T();
                    sp_p->eval_Ggtk_func = eval_bf(casadi::Function("eval_Ggtk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(Ggt)}, opts));
                }
                // lagraigian contribution of inequality constraints
                auto dual_ineq_sym = casadi::MX::sym("dual_ineq", ng_ineq);
                if (ng_ineq > 0)
                {
                    auto ineq_sym = sf_p->ineq({x_sym, u_sym, stage_params_sym, global_params_sym});
                    auto ineq = ineq_sym[0];
                    auto rq_ineq = casadi::MX::zeros(nu + ss.dims.nx, 1);
                    auto RSQlag = casadi::MX::hessian(casadi::MX::dot(dual_ineq_sym, ineq), ux, rq_ineq);
                    RSQ += RSQlag;
                    rq += rq_ineq;
                    sp_p->eval_gineqk_func = eval_bf(casadi::Function("eval_ineqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(ineq)}, opts));
                    auto Ggineqt = casadi::MX::zeros(nu + ss.dims.nx + 1, ng_ineq);
                    Ggineqt(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, ng_ineq)) = casadi::MX::jacobian(ineq, ux).T();
                    Ggineqt(nu + ss.dims.nx, casadi::Slice(0, ng_ineq)) = ineq.T();
                    sp_p->eval_Ggt_ineqk_func = eval_bf(casadi::Function("eval_Ggineqtk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(Ggineqt)}, opts));
                    // add inequality bounds
                    sp_p->lower.resize(ng_ineq);
                    sp_p->upper.resize(ng_ineq);
                    for (int j = 0; j < ng_ineq; j++)
                    {
                        sp_p->lower[j] = (double)sf_p->ineq_lb(j, 0);
                        sp_p->upper[j] = (double)sf_p->ineq_ub(j, 0);
                    }
                }
                // assemble RSQrqt
                auto RSQrqt = casadi::MX::zeros(nu + ss.dims.nx + 1, nu + ss.dims.nx);
                RSQrqt(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, nu + ss.dims.nx)) = RSQ;
                RSQrqt(nu + ss.dims.nx, casadi::Slice(0, nu + ss.dims.nx)) = rq.T();
                // prepare the function
                sp_p->eval_RSQrqtk_func = eval_bf(casadi::Function("eval_RSQrqtk", {x_sym, u_sym, stage_params_sym, global_params_sym, dual_dyn_sym, dual_eq_sym, dual_ineq_sym},
                                                                   {casadi::MX::densify(RSQrqt)}, opts));
            }
        };
        int eval_BAbtk(const double *states_kp1,
                       const double *inputs_k,
                       const double *states_k,
                       const double *stage_params_k,
                       const double *global_params,
                       MAT *res,
                       const int k) override
        {
            arg[0] = states_k;
            arg[1] = states_kp1;
            arg[2] = inputs_k;
            arg[3] = stage_params_k;
            arg[4] = global_params;
            eval_BAbtk_func(arg, res);
            return 0;
        };
        int eval_RSQrqtk(const double *objective_scale,
                         const double *inputs_k,
                         const double *states_k,
                         const double *lam_dyn_k,
                         const double *lam_eq_k,
                         const double *lam_ineq_k,
                         const double *stage_params_k,
                         const double *global_params,
                         MAT *res,
                         const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            arg[4] = lam_dyn_k;
            arg[5] = lam_eq_k;
            arg[6] = lam_ineq_k;
            if (k == 0)
                sp_initial.eval_RSQrqtk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_RSQrqtk_func(arg, res);
            else
                sp_middle.eval_RSQrqtk_func(arg, res);
            return 0;
        }
        int eval_Ggtk(
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            MAT *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_Ggtk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_Ggtk_func(arg, res);
            else
                sp_middle.eval_Ggtk_func(arg, res);
            return 0;
        }
        int eval_Ggt_ineqk(
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            MAT *res,
            const int k)
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_Ggt_ineqk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_Ggt_ineqk_func(arg, res);
            else
                sp_middle.eval_Ggt_ineqk_func(arg, res);
            return 0;
        }
        int eval_bk(
            const double *states_kp1,
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            double *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = states_kp1;
            arg[2] = inputs_k;
            arg[3] = stage_params_k;
            arg[4] = global_params;
            eval_bk_func(arg, res);
            return 0;
        }
        int eval_gk(
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            double *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_gk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_gk_func(arg, res);
            else
                sp_middle.eval_gk_func(arg, res);
            return 0;
        }
        int eval_gineqk(
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            double *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_gineqk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_gineqk_func(arg, res);
            else
                sp_middle.eval_gineqk_func(arg, res);
            return 0;
        }
        int eval_rqk(
            const double *objective_scale,
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            double *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_rqk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_rqk_func(arg, res);
            else
                sp_middle.eval_rqk_func(arg, res);
            return 0;
        }

        int eval_Lk(
            const double *objective_scale,
            const double *inputs_k,
            const double *states_k,
            const double *stage_params_k,
            const double *global_params,
            double *res,
            const int k) override
        {
            arg[0] = states_k;
            arg[1] = inputs_k;
            arg[2] = stage_params_k;
            arg[3] = global_params;
            if (k == 0)
                sp_initial.eval_Lk_func(arg, res);
            else if (k == K_ - 1)
                sp_terminal.eval_Lk_func(arg, res);
            else
                sp_middle.eval_Lk_func(arg, res);
            return 0;
        }
        int get_initial_xk(double *xk, const int k) const override
        {
            return 0;
        };
        int get_initial_uk(double *uk, const int k) const override
        {
            return 0;
        };
        int set_initial_xk(double *xk, const int k)
        {
            return 0;
        };
        int set_initial_uk(double *uk, const int k)
        {
            return 0;
        };
        int get_boundsk(double *lower, double *upper, const int k) const override
        {
            int ngineq = this->get_ng_ineq_k(k);
            const double *lower_ineq;
            const double *upper_ineq;
            if (k == 0)
            {
                lower_ineq = sp_initial.lower.data();
                upper_ineq = sp_initial.upper.data();
            }
            else if (k == K_ - 1)
            {
                lower_ineq = sp_terminal.lower.data();
                upper_ineq = sp_terminal.upper.data();
            }
            else
            {
                lower_ineq = sp_middle.lower.data();
                upper_ineq = sp_middle.upper.data();
            }

            for (int j = 0; j < ngineq; j++)
            {
                lower[j] = lower_ineq[j];
                upper[j] = upper_ineq[j];
            }
            return 0;
        };
        int get_default_stage_paramsk(double *stage_params_res, const int k) const override
        {
            return 0;
        }
        int get_default_global_params(double *global_params_res) const override
        {
            return 0;
        }

    private:
        class eval_bf
        {
        public:
            eval_bf(){};
            eval_bf(const casadi::Function &funcin)
            {
                m = (int)funcin.size1_out(0);
                n = (int)funcin.size2_out(0);
                func = funcin;
                mem = 0;
                // allocate work vectors
                size_t sz_arg,
                    sz_res, sz_iw, sz_w;
                sz_arg = func.n_in();
                sz_res = func.n_out();
                func.sz_work(sz_arg, sz_res, sz_iw, sz_w);
                iw.resize(sz_iw);
                w.resize(sz_w);
                bufout.resize(func.nnz_out(0));
                bufdata.resize(sz_res);
                resdata.resize(sz_res);
                argdata.resize(sz_arg);
                n_in = func.n_in();

                // resdata = {nullptr};
                dirty = false;
            }
            void operator=(const eval_bf &other)
            {
                // copy all member values
                m = other.m;
                n = other.n;
                n_in = other.n_in;
                func = other.func;
                mem = other.mem;
                bufout = other.bufout;
                bufdata = other.bufdata;
                resdata = other.resdata;
                argdata = other.argdata;
                iw = other.iw;
                w = other.w;
                dirty = other.dirty;
            }
            // copy operator
            void operator()(std::vector<const double *> &arg, MAT *res)
            {
                if (dirty)
                    return;
                // inputs
                for (int j = 0; j < n_in; j++)
                    argdata[j] = arg[j];
                // outputs
                bufdata[0] = bufout.data();
                func(argdata.data(), bufdata.data(), iw.data(), w.data(), 0);
                // func(arg, bufdata);
                PACKMAT(m, n, bufout.data(), m, res, 0, 0);
            }
            void operator()(std::vector<const double *> &arg, double *res)
            {
                if (dirty)
                    return;
                // inputs
                for (int j = 0; j < n_in; j++)
                    argdata[j] = arg[j];
                // outputs
                resdata[0] = res;
                func(argdata.data(), resdata.data(), iw.data(), w.data(), 0);
                // func(arg, resdata);
            }
            // ~eval_bf() { func.release(mem); }
            int m;
            int n;
            int mem;
            int n_in;
            std::vector<double> bufout;
            std::vector<double *> bufdata;
            std::vector<double *> resdata;
            std::vector<const double *> argdata;
            std::vector<long long int> iw;
            std::vector<double> w;
            casadi::Function func;
            bool dirty = true;
        };
        eval_bf eval_BAbtk_func; // OK
        eval_bf eval_bk_func;    // OK
        struct stageproperties
        {
            eval_bf eval_Lk_func;      // OK
            eval_bf eval_RSQrqtk_func; // OK
            eval_bf eval_rqk_func;     // OK
            eval_bf eval_Ggtk_func;
            eval_bf eval_gk_func;
            eval_bf eval_Ggt_ineqk_func;
            eval_bf eval_gineqk_func;
            std::vector<double> lower;
            std::vector<double> upper;
        };
        stageproperties sp_initial;
        stageproperties sp_middle;
        stageproperties sp_terminal;
        std::vector<const double *> arg;
    };

} // namespace fatrop_casadi