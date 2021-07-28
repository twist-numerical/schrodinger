#ifndef SCHRODINGER2D_DOMAIN_H
#define SCHRODINGER2D_DOMAIN_H

#include <array>
#include <vector>
#include <cmath>
#include "util/polymorphic_value.h"
#include "util/geometry.h"

template<class Scalar, int d>
class Domain {
public:
    virtual bool contains(const Vector<Scalar, d> &point) const = 0;

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(const Ray<Scalar, d> &) const = 0;

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

    virtual bool contains(const Vector<Scalar, d> &point) const override {
        for (auto &dom : subDomains)
            if (dom->contains(point))
                return true;
        return false;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(const Ray<Scalar, d> &ray) const override {
        typedef struct {
            Scalar point;
            bool isStart;
        } Point;
        std::vector<Point> points;
        for (auto &dom : subDomains)
            for (auto &sec : dom->intersections(ray)) {
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

    virtual bool contains(const Vector<Scalar, d> &point) const override {
        for (int i = 0; i < d; ++i)
            if (point[i] <= bounds[2 * i] || point[i] >= bounds[2 * i + 1])
                return false;
        return true;
    }

    bool withinSide(const Vector<Scalar, d> &point, int axis) const {
        for (int i = 0; i < d; ++i)
            if (i != axis && (point[i] <= bounds[2 * i] || point[i] >= bounds[2 * i + 1]))
                return false;
        return true;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(const Ray<Scalar, d> &ray) const override {
        typedef Vector<Scalar, d> V;
        Scalar low = std::numeric_limits<Scalar>::infinity();
        Scalar high = -low;

        V min = this->min();
        V max = this->max();

        bool intersecting = false;
        for (int i = 0; i < d; ++i) {
            for (const V &corner: {min, max}) {
                Scalar a = ray.cast({corner, V::Unit(i)});
                if (withinSide(ray(a), i)) {
                    low = std::min(a, low);
                    high = std::max(a, high);
                    intersecting = true;
                }
            }
        }
        if (!intersecting || high - low < 1e-9)
            return {};
        return {{low, high}};
    }

    Vector<Scalar, d> min() const {
        Vector<Scalar, d> v;
        for (int i = 0; i < d; ++i)
            v[i] = min(i);
        return v;
    }

    Vector<Scalar, d> max() const {
        Vector<Scalar, d> v;
        for (int i = 0; i < d; ++i)
            v[i] = max(i);
        return v;
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

template<class Scalar, int d>
class Sphere : public Domain<Scalar, d> {
private:
    Scalar radius;
    Vector<Scalar, d> center;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit constexpr Sphere(Scalar radius = 1) : radius(radius), center(Vector<Scalar, d>::Zero()) {};

    explicit constexpr Sphere(const Vector<Scalar, d> &center, Scalar radius = 1)
            : radius(radius), center(center) {};

    virtual bool contains(const Vector<Scalar, d> &point) const override {
        Vector<Scalar, d> p = point - center;
        return p.dot(p) < radius * radius;
    }

    virtual std::vector<std::pair<Scalar, Scalar>>
    intersections(const Ray<Scalar, d> &ray) const override {
        typedef Vector<Scalar, d> V;

        V o = ray.origin - center;
        Scalar a = ray.direction.dot(ray.direction), b = 2 * o.dot(ray.direction), c = o.dot(o) - radius * radius;
        Scalar D = b * b - 4 * a * c;
        if (D < 1e-12)
            return {};
        D = std::sqrt(D);
        a *= 2;
        return {{(-b - D) / a, (-b + D) / a}};
    }

    virtual Scalar min(int axis) const override {
        return center[axis] - radius;
    }

    virtual Scalar max(int axis) const override {
        return center[axis] + radius;
    }

    virtual Sphere *clone() const override {
        return new Sphere(*this);
    };
};

#endif //SCHRODINGER2D_DOMAIN_H
