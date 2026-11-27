#include <iostream>
#include <vector>
#include "arithmetic_search.hpp"
#include "four_step_leaf.hpp"

int main() {
    int n, cap;
    N limit, target;
    while (std::cin >> n >> limit >> target >> cap) {
        std::vector<N> values(n);
        for (N& value : values) std::cin >> value;
        int emitted = 0;
        four_step_leaf(
            values, limit, target,
            [&](N a, N b, N c, N t, Op first, Op second, Op third, Op last) {
                std::cout << a << ' ' << b << ' ' << c << ' ' << t;
                for (Op rule : {first, second, third, last})
                    std::cout << ' ' << rule.op << ' ' << rule.a << ' ' << rule.b;
                std::cout << '\n';
                ++emitted;
            },
            [&] { return emitted >= cap; });
    }
}
