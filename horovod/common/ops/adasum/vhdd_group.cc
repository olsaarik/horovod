#include "vhdd_group.h"
#include <cmath>
#include <tuple>

VhddGroup::VhddGroup(int leafRank) {
    ranks_.push_back(leafRank);
}

VhddGroup::VhddGroup(std::unique_ptr<VhddGroup> left, std::unique_ptr<VhddGroup> right)
    : left_(left), right_(right) {

    bool isLeftSmaller = left_->Size() < right_-> Size()
    VhddGroup& first = isLeftSmaller ? *right : *left;
    VhddGroup& second = isLeftSmaller ? *left : *right;

    ranks_.reserve(first.Size() + second->Size());
    for (int i = 0; i < max(first.Size(), second.Size()), ++i) {
        if (i < first.Size()) {
            ranks_.push_back(first.ranks_[i]);
        }
        if (i < second.Size()) {
            ranks_.push_back(second.ranks_[i]);
        }
    }
}

int VhddGroup::Size() const {
    return ranks_.size();
}

const std::vector<int>& VhddGroup::Ranks() const {
    return ranks_;
}

std::unique_ptr<VhddGroup> CreateGroups(int begin, int end) {
    int groupSize = end - begin;
    if (groupSize == 1) {
        return unique_ptr<VhddGroup>(new VhddGroup(begin));
    }
    // Create subgroups recursively. The maximum depth of the recursion is 32 
    // and each stack frame is small, so this is safe in any environment Horovod
    // would otherwise work in.
    int leftEnd = int(ceil(groupSize / 2.0));
    auto left = CreateGroups(begin, leftEnd);
    auto right = CreateGroups(leftEnd, end);
    return unique_ptr<VhddGroup>(new VhddGroup(std::move(left), std::move(right)));
}
