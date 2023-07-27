#pragma once
#include "single_stage.hpp"
#include "single_stage_fatrop.hpp"
#include <vector>
#include <memory>
#include <ocp/StageOCP.hpp>

namespace fatrop_casadi
{

    class MultiStage
    {
        std::shared_ptr<SingleStage> stage(const int K)
        {
            auto stage = std::make_shared<SingleStage>(K);
            stages.push_back(stage);
            if (stages.size() != 0)
            {
                link_constraints.push_back(casadi::MX::zeros(0, 0));
            }
            return stage;
        }
        void connect_stages(const casadi::MX &x, const casadi::MX &y)
        {
            // determine from which stage variable x and y are
            int x_stage = -1;
            int y_stage = -1;
            for (int i = 0; i < stages.size(); i++)
            {
                if (stages[i]->depends_on(x))
                {
                    if (x_stage != -1)
                    {
                        throw std::runtime_error("x is dependent on multiple stages");
                    }
                    x_stage = i;
                }
                if (stages[i]->depends_on(y))
                {
                    if (y_stage != -1)
                    {
                        throw std::runtime_error("y is dependent on multiple stages");
                    }
                    y_stage = i;
                }
                if (x_stage == -1 && y_stage == -1)
                {
                    throw std::runtime_error("x and y are not dependent on any stage");
                }
                if (y_stage != x_stage + 1)
                {
                    throw std::runtime_error("x and y are not consecutive stages");
                }
                link_constraints[x_stage] = casadi::MX::vertcat({link_constraints[x_stage], casadi::MX::veccat({x - y})});
            }
        }

    public:
        std::vector<std::shared_ptr<SingleStage>> stages;
        std::vector<casadi::MX> link_constraints;
    };

    class MultiStageFatropAdapter: public fatrop::OCPAbstract
    {
        MultiStageFatropAdapter(MultiStage &multi_stage)
        {
            // initialize the single stages
            for (auto &stage : multi_stage.stages)
            {
                single_stage_adapters.push_back(SingleStageFatropAdapter(*stage));
            }
        }
        std::vector<SingleStageFatropAdapter> single_stage_adapters;
    };
} // namespace fatrop_casadi