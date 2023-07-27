#pragma once
#include <casadi/casadi.hpp>

namespace fatrop_casadi
{
    struct comp_mx
    {
        bool operator()(const casadi::MX &a, const casadi::MX &b) const
        {
            return a.get() < b.get();
        }
    };

    template <typename Tinternal>
    class SharedObj : public std::shared_ptr<Tinternal>
    {
    public:
        template <class... args>
        SharedObj(args &&...a) : std::shared_ptr<Tinternal>(std::make_shared<Tinternal>(std::forward<args>(a)...)){};
        operator Tinternal &() { return *this->get(); };
    };
}