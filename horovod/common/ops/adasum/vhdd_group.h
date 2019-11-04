class VhddGroup {
public:
    VhddGroup(std::vector<int> ranks);
    VhddGroup(std::unique_ptr<VhddGroup> left, std::unique_ptr<VhddGroup> right);

    int Size() const;
    const std::vector<int>& Ranks() const;
private:
    std::vector<int> ranks_;
    std::unique_ptr<VhddGroup> left_;
    std::unique_ptr<VhddGroup> right_;
};
