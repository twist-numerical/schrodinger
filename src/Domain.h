#ifndef SCHRODINGER2D_DOMAIN_H
#define SCHRODINGER2D_DOMAIN_H

#include <array>
#include <vector>
#include <cmath>
#include "util/polymorphic_value.h"

template<class Scalar, int d>
class Domain {
public:
    virtual bool contains(const std::array<Scalar, d> &point) const = 0;

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(int axis, const std::array<Scalar, d> &point) const = 0;

    virtual Scalar min(int axis) const = 0;

    virtual Scalar max(int axis) const = 0;

    virtual Domain *clone() const = 0;

    virtual ~Domain() {};

    struct copy {
        Domain *operator()(const Domain &t) const { return t.clone(); }
    };
};

template<class Scalar, int d>
class Union : public Domain<Scalar, d> {
public:
    std::vector<isocpp_p0201::polymorphic_value<Domain<Scalar, d>>> subDomains;

    Union(const std::vector<const Domain<Scalar, d> *> &domains) {
        for (auto &dom : domains) {
            subDomains.push_back(isocpp_p0201::polymorphic_value<Domain<Scalar, d>>(
                    static_cast<Domain<Scalar, d> *>(dom->clone()),
                    typename Domain<Scalar, d>::copy{}));
        }
    }

    template<class... Ts>
    Union(const Ts &... ts) {
        subDomains = {
                isocpp_p0201::polymorphic_value<Domain<Scalar, d>>(
                        static_cast<Domain<Scalar, d> *>(ts.clone()), typename Domain<Scalar, d>::copy{})...
        };
    }

    virtual bool contains(const std::array<Scalar, d> &point) const override {
        for (auto &dom : subDomains)
            if (dom->contains(point))
                return true;
        return false;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(int axis, const std::array<Scalar, d> &point) const override {
        typedef struct {
            Scalar point;
            bool isStart;
        } Point;
        std::vector<Point> points;
        for (auto &dom : subDomains)
            for (auto &sec : dom->intersections(axis, point)) {
                points.push_back({.point=sec.first, .isStart=true});
                points.push_back({.point=sec.second, .isStart=false});
            }
        std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
            if (a.point == b.point)
                return a.isStart && !b.isStart;
            return a.point < b.point;
        });
        std::vector<std::pair<Scalar, Scalar>> intersections;
        int nested = 0;
        for (auto &p : points) {
            if (p.isStart) {
                if (nested++ == 0)
                    intersections.template emplace_back(p.point, 0);
            } else {
                if (--nested == 0)
                    intersections.back().second = p.point;
            }
        }
        return intersections;
    }

    virtual Scalar min(int axis) const override {
        Scalar min = Scalar(std::numeric_limits<float>::infinity());
        for (const auto &dom : subDomains)
            min = std::min(min, dom->min(axis));
        return min;
    }

    virtual Scalar max(int axis) const override {
        Scalar max = -Scalar(std::numeric_limits<float>::infinity());
        for (const auto &dom : subDomains)
            max = std::max(max, dom->max(axis));
        return max;
    };

    virtual Union<Scalar, d> *clone() const override {
        return new Union<Scalar, d>(*this);
    };
};

template<class Scalar, int d>
class Rectangle : public Domain<Scalar, d> {
private:
    std::array<Scalar, d * 2> bounds;
public:
    constexpr Rectangle() {};

    template<typename... T, typename=typename std::enable_if<
            sizeof...(T) == 2 * d && (std::is_convertible<T, Scalar>::value && ...)>::type>
    constexpr Rectangle(T... bounds) : bounds({bounds...}) {}

    virtual bool contains(const std::array<Scalar, d> &point) const override {
        for (int i = 0; i < d; ++i)
            if (point[i] <= bounds[2 * i] || point[i] >= bounds[2 * i + 1])
                return false;
        return true;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(int axis, const std::array<Scalar, d> &point) const override {
        for (int i = 0; i < d; ++i)
            if (i != axis && (point[i] <= bounds[2 * i] || point[i] >= bounds[2 * i + 1]))
                return {};
        return {{bounds[2 * axis], bounds[2 * axis + 1]}};
    }

    virtual Scalar min(int axis) const override {
        return bounds[2 * axis];
    }

    virtual Scalar max(int axis) const override {
        return bounds[2 * axis + 1];
    }

    virtual Rectangle *clone() const override {
        return new Rectangle(*this);
    };
};

template<class Scalar>
class Circle : public Domain<Scalar, 2> {
private:
    Scalar radius = 1;
    std::array<Scalar, 2> center = {0, 0};
public:
    explicit constexpr Circle(Scalar radius = 1) : radius(radius), center({0, 0}) {};

    explicit constexpr Circle(const std::array<Scalar, 2> &center, Scalar radius = 1)
            : radius(radius), center(center) {};

    virtual bool contains(const std::array<Scalar, 2> &point) const override {
        Scalar dx = point[0] - center[0], dy = point[1] - center[1];
        return dx * dx + dy * dy < radius * radius;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(int axis, const std::array<Scalar, 2> &point) const override {
        double p = point[1 - axis] - center[1 - axis];
        if (std::abs(p) >= radius)
            return {};

        Scalar s = std::sqrt(radius * radius - p * p);
        return {{center[axis] - s, center[axis] + s}};
    }

    virtual Scalar min(int axis) const override {
        return center[axis] - radius;
    }

    virtual Scalar max(int axis) const override {
        return center[axis] + radius;
    }

    virtual Circle *clone() const override {
        return new Circle(*this);
    };
};

#endif //SCHRODINGER2D_DOMAIN_H
