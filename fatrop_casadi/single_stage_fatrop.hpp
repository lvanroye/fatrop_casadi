
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
#include "single_stage.hpp"
#define JIT_HACKED_CASADI 0
namespace fatrop_casadi
{
    typedef int (*eval_t)(const double **arg, double **res,
                          long long int *iw, double *w, int);

    class SingleStageFatropAdapter : public fatrop::StageOCP
    {
    public:
        SingleStageFatropAdapter(SingleStage &ss, const casadi::Dict &opts = {}) : fatrop::StageOCP(ss.dims.nu, ss.dims.nx, ss.dims.ngI, ss.dims.ng, ss.dims.ngF, ss.dims.ng_ineqI, ss.dims.ng_ineq, ss.dims.ng_ineqF, ss.dims.n_stage_params, ss.dims.n_global_params, ss.dims.K), arg(10)
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
            eval_BAbtk_func = eval_bf(sx_func_helper(casadi::Function("eval_BAbtk", {x_sym, xp1_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(BAbt)}), opts));
            eval_bk_func = eval_bf(sx_func_helper(casadi::Function("eval_bk", {x_sym, xp1_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(b)}), opts));
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
                sp_p->eval_Lk_func = eval_bf(sx_func_helper(casadi::Function("eval_Lk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(obj)}), opts));
                casadi::MX rq;
                auto RSQ = casadi::MX::hessian(obj, ux, rq);
                sp_p->eval_rqk_func = eval_bf(sx_func_helper(casadi::Function("eval_rqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(rq)}), opts));
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
                    sp_p->eval_gk_func = eval_bf(sx_func_helper(casadi::Function("eval_eqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(eq)}), opts));
                    auto Ggt = casadi::MX::zeros(nu + ss.dims.nx + 1, ng);
                    Ggt(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, ng)) = casadi::MX::jacobian(eq, ux).T();
                    Ggt(nu + ss.dims.nx, casadi::Slice(0, ng)) = eq.T();
                    sp_p->eval_Ggtk_func = eval_bf(sx_func_helper(casadi::Function("eval_Ggtk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(Ggt)}), opts));
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
                    sp_p->eval_gineqk_func = eval_bf(sx_func_helper(casadi::Function("eval_ineqk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(ineq)}), opts));
                    auto Ggineqt = casadi::MX::zeros(nu + ss.dims.nx + 1, ng_ineq);
                    Ggineqt(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, ng_ineq)) = casadi::MX::jacobian(ineq, ux).T();
                    Ggineqt(nu + ss.dims.nx, casadi::Slice(0, ng_ineq)) = ineq.T();
                    sp_p->eval_Ggt_ineqk_func = eval_bf(sx_func_helper(casadi::Function("eval_Ggineqtk", {x_sym, u_sym, stage_params_sym, global_params_sym}, {casadi::MX::densify(Ggineqt)}), opts));
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
                sp_p->eval_RSQrqtk_func = eval_bf(sx_func_helper(casadi::Function("eval_RSQrqtk", {x_sym, u_sym, stage_params_sym, global_params_sym, dual_dyn_sym, dual_eq_sym, dual_ineq_sym},
                                                                                  {casadi::MX::densify(RSQrqt)}),
                                                                 opts));
                // prepare BFGS update function
                auto RSQrqk = casadi::MX::sym("RSQrqk", nu + ss.dims.nx + 1, nu + ss.dims.nx);
                auto Bk = RSQrqk(casadi::Slice(0, nu + ss.dims.nx), casadi::Slice(0, nu + ss.dims.nx));
                auto ux_prev = casadi::MX::sym("ux_prev", nu + ss.dims.nx);
                // auto rq_prev = RSQrqk(nu + ss.dims.nx, casadi::Slice(0, nu + ss.dims.nx)).T();
                auto rq_prev = casadi::MX::substitute(rq, ux, ux_prev);               
                auto Bkp1 = update_bfgs(Bk, ux_prev, ux, rq_prev, rq);
                auto RSQrq_bfgs = casadi::MX::vertcat({Bkp1, rq.T()});
                sp_p->eval_update_bfgs_func = eval_bf(sx_func_helper(casadi::Function("eval_update_bfgs", {ux_prev, RSQrqk, x_sym, u_sym, stage_params_sym, global_params_sym, dual_dyn_sym, dual_eq_sym, dual_ineq_sym},
                                                                                      {casadi::MX::densify(RSQrq_bfgs)}),
                                                                     opts));
            }
            reset_bfgs();
        };
        casadi::MX update_bfgs(casadi::MX &Bk, casadi::MX &xk, casadi::MX &xkp1, casadi::MX &gradk, casadi::MX &gradkp1)
        {
            auto sk = xkp1 - xk;
            auto sts = casadi::MX::dot(sk, sk);
            auto yk = gradkp1 - gradk;
            auto sty = casadi::MX::dot(yk, sk);
            auto vk = casadi::MX::mtimes(Bk, sk);
            auto stv = casadi::MX::dot(vk, sk);
            auto beta = -1.0 / stv;
            auto theta_k = casadi::MX::if_else(sty > 0.2*stv, 1.0, 0.8*stv / (stv - sty));
            auto yk_tilde = theta_k * yk + (1.0 - theta_k) * vk;
            auto sty_tilde = casadi::MX::dot(yk_tilde, sk);
            auto alpha_tilde = 1.0 / sty_tilde;
            auto Bkp1 = Bk + alpha_tilde * casadi::MX::mtimes(yk_tilde, yk_tilde.T()) + beta * casadi::MX::mtimes(vk, vk.T());
            Bkp1 = casadi::MX::if_else(sty !=0.0, Bkp1 , Bk);
            return Bkp1;
        }
        void reset_bfgs()
        {
            RSQrq_bfgs_buf.resize(K_);
            RSQrq_bfgs_temp.resize(K_);
            ux_buff.resize(K_);
            grad_buf.resize(K_);
            for (int k = 0; k < K_; k++)
            {
                int nu = this->get_nuk(k);
                int nx = this->get_nxk(k);
                ux_buff[k] = std::vector<double>(nu + nx, 0.0);
                // cpy initial value to ux
                grad_buf[k] = std::vector<double>(nu + nx, 0.0);
                RSQrq_bfgs_temp[k] = std::vector<double>((nu + nx + 1) * (nu + nx), 0.0);
                RSQrq_bfgs_buf[k] = std::vector<double>((nu + nx + 1) * (nu + nx), 0.0);
                set_eye(nu + nx + 1, nu + nx, nu + nx, RSQrq_bfgs_buf[k]);
            }
        }
        void set_eye(const int m, const int n, const int l, std::vector<double> &vec)
        {

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    vec[i + m * j] = (i == j && i < l) ? 1e0 : 0.0;
                }
            }
        }
        void set_zero(const int m, const int n, std::vector<double> &vec)
        {

            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < m; i++)
                {
                    vec[i + m * j] = 0.0;
                }
            }
        }
        // void DM_to_raw(const casadi::DM &in, std::vector<double> &out)
        // {
        //     casadi::DM in_dense = casadi::DM::densify(in);
        //     // casadi uses column major order
        //     out.resize(in_dense.size1() * in_dense.size2());
        //     for (size_t i = 0; i < in_dense.size2(); i++)
        //     {
        //         for (size_t j = 0; j < in_dense.size1(); j++)
        //         {
        //             out[i * in_dense.size1() + j] = in_dense(j, i).scalar();
        //         }
        //     }
        // }
        // double dist_inf(const int m, const double *v1, const double *v2)
        // {
        //     double res = 0.0;
        //     for (int i = 0; i < m; i++)
        //     {
        //         res = std::max(res, std::abs(v1[i] - v2[i]));
        //     }
        //     return res;
        // }
        // bool is_zero(const int m, const double *v1)
        // {
        //     for (int i = 0; i < m; i++)
        //     {
        //         if (v1[i] != 0.0)
        //             return false;
        //     }
        //     return true;
        // }
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
        bool bfgs = false;
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
            if (bfgs)
            {

                int nu = this->get_nuk(k);
                int nx = this->get_nxk(k);
                arg[0] = ux_buff[k].data();
                arg[1] = RSQrq_bfgs_buf[k].data();
                arg[2] = states_k;
                arg[3] = inputs_k;
                arg[4] = stage_params_k;
                arg[5] = global_params;
                arg[6] = lam_dyn_k;
                arg[7] = lam_eq_k;
                arg[8] = lam_ineq_k;
                if (k == 0)
                    sp_initial.eval_update_bfgs_func(arg, res, RSQrq_bfgs_temp[k].data());
                else if (k == K_ - 1)
                    sp_terminal.eval_update_bfgs_func(arg, res, RSQrq_bfgs_temp[k].data());
                else
                    sp_middle.eval_update_bfgs_func(arg, res, RSQrq_bfgs_temp[k].data());

                // copy elements from RSQrq_bfgs temp to RSQrq_bfgs_buf
                for (int i = 0; i < RSQrq_bfgs_temp[k].size(); i++)
                    RSQrq_bfgs_buf[k][i] = RSQrq_bfgs_temp[k][i];

                // save ux
                for (int i = 0; i < nu; i++)
                    ux_buff[k][i] = inputs_k[i];
                for (int i = 0; i < nx; i++)
                    ux_buff[k][nu + i] = states_k[i];
                // blasfeo_print_dmat(nu+nx+1, nu+nx, res, 0,0);
                return 0;
            }
            else
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
        casadi::Function sx_func_helper(const casadi::Function &func_in, const casadi::Dict &opts)
        {
            int n_in = func_in.n_in();
            int n_out = func_in.n_out();
            casadi::SXVector sx_in(n_in);
            for (int i = 0; i < n_in; i++)
            {
                sx_in[i] = casadi::SX::sym(func_in.name_in(i), func_in.size1_in(i), func_in.size2_in(i));
            }
            auto out_sx = func_in(sx_in);
            return casadi::Function(func_in.name(), sx_in, out_sx, opts);
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
                fast_jit();

                // resdata = {nullptr};
                dirty = false;
            }
            void fast_jit()
            {
                casadi::FunctionInternal *func_internal = func.get();
                jit_name_ = func.name();
                jit_name_ = casadi::temporary_file(jit_name_, ".c");
                jit_name_ = std::string(jit_name_.begin(), jit_name_.begin() + jit_name_.size() - 2);
                if (func_internal->has_codegen())
                {
                    // this part is based on casadi/core/function_internal.cpp -- all rights reserved to the original authors
                    // JIT everything
                    casadi::Dict opts;
                    // Override the default to avoid random strings in the generated code
                    opts["prefix"] = "jit";
                    casadi::CodeGenerator gen(jit_name_, opts);
                    gen.add(func);
                    jit_options_ = casadi::Dict({{"flags", "-Ofast -march=native -ffast-math"}});
                    jit_directory = casadi::get_from_dict(jit_options_, "directory", std::string(""));
                    std::string compiler_plugin_ = "shell";
                    compiler_ = casadi::Importer(gen.generate(jit_directory), compiler_plugin_, jit_options_);
                    eval_ = (eval_t)compiler_.get_function(func.name());
                    casadi_assert(eval_ != nullptr, "Cannot load JIT'ed function.");
                    compiled_jit = true;
                }
                else
                {
                    std::cout << "jit compilation not possible for the provided functions" << std::endl;
                }
            }
            std::string jit_name_;
            casadi::Dict jit_options_;
            std::string jit_directory;
            bool compiled_jit = false;
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
                fast_jit();
                // eval_ = other.eval_;
                // other.eval_ = nullptr;
            }
            void operator()(std::vector<const double *> &arg, MAT *res, double *buff)
            {
                if (dirty)
                    return;
                // inputs
                for (int j = 0; j < n_in; j++)
                    argdata[j] = arg[j];
                // outputs
                bufdata[0] = buff;
#ifdef JIT_HACKED_CASADI
                eval_(argdata.data(), bufdata.data(), iw.data(), w.data(), 0);
#else
                func(argdata.data(), bufdata.data(), iw.data(), w.data(), 0);
#endif
                // func(arg, bufdata);
                PACKMAT(m, n, buff, m, res, 0, 0);
            }
            // copy operator
            void operator()(std::vector<const double *> &arg, MAT *res)
            {
                this->operator()(arg, res, bufout.data());
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
#ifdef JIT_HACKED_CASADI
                eval_(argdata.data(), resdata.data(), iw.data(), w.data(), 0);
#else
                func(argdata.data(), resdata.data(), iw.data(), w.data(), 0);
#endif
                // func(arg, resdata);
            }
            ~eval_bf()
            {
                if (compiled_jit)
                {
                    std::string jit_directory = get_from_dict(jit_options_, "directory", std::string(""));
                    std::string jit_name = jit_directory + jit_name_ + ".c";
                    if (remove(jit_name.c_str()))
                        casadi_warning("Failed to remove " + jit_name);
                }
            }
            int m;
            int n;
            int mem;
            int n_in;
            eval_t eval_;
            casadi::Importer compiler_;
            std::vector<double> bufout;
            std::vector<double *> bufdata;
            std::vector<double *> resdata;
            std::vector<const double *> argdata;
            std::vector<long long int> iw;
            std::vector<double> w;
            casadi::Function func;
            bool dirty = true;
            bool bfgs = true;
        };
        std::vector<std::vector<double>> RSQrq_bfgs_buf;
        std::vector<std::vector<double>> RSQrq_bfgs_temp;
        std::vector<std::vector<double>> grad_buf;
        std::vector<std::vector<double>> ux_buff;
        eval_bf eval_BAbtk_func; // OK
        eval_bf eval_bk_func;    // OK
        struct stageproperties
        {
            eval_bf eval_Lk_func;          // OK
            eval_bf eval_RSQrqtk_func;     // OK
            eval_bf eval_update_bfgs_func; // OK
            eval_bf eval_rqk_func;         // OK
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
}