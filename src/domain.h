#ifndef STRANDS_DOMAIN_H
#define STRANDS_DOMAIN_H

#include <array>
#include <vector>
#include <cmath>
#include "util/geometry.h"

namespace strands::geometry {

    template<class Scalar, int d>
    class Domain {
    public:
        virtual bool contains(const Vector<Scalar, d> &point) const = 0;

        virtual std::vector<std::pair<Scalar, Scalar>> intersections(const Ray<Scalar, d> &) const = 0;

        virtual std::pair<Scalar, Scalar> bounds(const Vector<Scalar, d> &direction) const = 0;

        virtual ~Domain() {};

        template<typename T>
        static std::shared_ptr<Domain<Scalar, d>> as_ptr(T t) {
            if constexpr (std::is_base_of_v<Domain<Scalar, d>, T>) {
                return std::make_shared<T>(t);
            } else if constexpr (std::is_convertible_v<T, std::shared_ptr<Domain<Scalar, d>>>) {
                return t;
            } else {
                static_assert(!sizeof(T *)); // == static_assert(false)
            }
        }
    };

    template<class Scalar, int d>
    class DomainTransform : public Domain<Scalar, d> {
    private:
        Eigen::Transform<Scalar, d, Eigen::Affine> transform;
        Eigen::Transform<Scalar, d, Eigen::Affine> invTransform;
        std::shared_ptr<const Domain<Scalar, d>> domain;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        template<class DomainType, class TransformType>
        DomainTransform(DomainType domain_, const TransformType &transform_) {
            domain = Domain<Scalar, 2>::as_ptr(domain_);
            transform = transform_;
            invTransform = transform.inverse();
        }

        virtual bool contains(const Vector<Scalar, d> &point) const override {
            return domain->contains(invTransform * point);
        };

        virtual std::vector<std::pair<Scalar, Scalar>>
        intersections(const Ray<Scalar, d> &ray) const override {
            return domain->intersections({invTransform * ray.origin, invTransform.linear() * ray.direction});
        }

        virtual std::pair<Scalar, Scalar> bounds(const Vector<Scalar, d> &direction) const override {
            Scalar shift = direction.dot(transform.translation()) / direction.dot(direction);
            Vector<Scalar, d> dir = invTransform.linear() * direction;
            Scalar min, max;
            std::tie(min, max) = domain->bounds(dir);
            return {min + shift, max + shift};
        }
    };

    template<class Scalar, int d>
    class Union : public Domain<Scalar, d> {
    public:
        std::vector<std::shared_ptr<Domain<Scalar, d>>> subdomains;

        Union(const std::vector<std::shared_ptr<Domain<Scalar, d>>> &domains) : subdomains(domains) {
        }

        template<class... Ts>
        explicit Union(const Ts &... ts) {
            subdomains = {
                    Domain<Scalar, 2>::as_ptr(ts)...
            };
        }

        virtual bool contains(const Vector<Scalar, d> &point) const override {
            for (auto &dom: subdomains)
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
            for (auto &dom: subdomains)
                for (auto &sec: dom->intersections(ray)) {
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
            for (auto &p: points) {
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

        virtual std::pair<Scalar, Scalar> bounds(const Vector<Scalar, d> &direction) const override {
            Scalar min = std::numeric_limits<Scalar>::infinity();
            Scalar max = -std::numeric_limits<Scalar>::infinity();
            Scalar b_min, b_max;

            for (const auto &dom: subdomains) {
                std::tie(b_min, b_max) = dom->bounds(direction);
                min = std::min(min, b_min);
                max = std::max(max, b_max);
            }

            return {min, max};
        }
    };

    template<class Scalar, int d>
    class Rectangle : public Domain<Scalar, d> {
    private:
        std::array<Scalar, d * 2> bounds_;
    public:
        constexpr Rectangle() {};

        template<typename... T, typename=typename std::enable_if<
                sizeof...(T) == 2 * d && (std::is_convertible<T, Scalar>::value && ...)>::type>
        constexpr Rectangle(T... bounds) : bounds_({bounds...}) {}

        template<int axis>
        Scalar &min() {
            return bounds_[2 * axis];
        }

        template<int axis>
        Scalar &max() {
            return bounds_[2 * axis + 1];
        }

        template<int axis>
        const Scalar &min() const {
            return bounds_[2 * axis];
        }

        template<int axis>
        const Scalar &max() const {
            return bounds_[2 * axis + 1];
        }

        virtual bool contains(const Vector<Scalar, d> &point) const override {
            for (int i = 0; i < d; ++i)
                if (point[i] <= bounds_[2 * i] || point[i] >= bounds_[2 * i + 1])
                    return false;
            return true;
        }

        bool withinSide(const Vector<Scalar, d> &point, int axis) const {
            for (int i = 0; i < d; ++i)
                if (i != axis && (point[i] <= bounds_[2 * i] || point[i] >= bounds_[2 * i + 1]))
                    return false;
            return true;
        }

        virtual std::vector<std::pair<Scalar, Scalar>>
        intersections(const Ray<Scalar, d> &ray) const override {
            typedef Vector<Scalar, d> V;
            Scalar low = std::numeric_limits<Scalar>::infinity();
            Scalar high = -low;

            V min, max;
            for (int i = 0; i < d; ++i) {
                min[i] = bounds_[2 * i];
                max[i] = bounds_[2 * i + 1];
            }

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

        virtual std::pair<Scalar, Scalar> bounds(const Vector<Scalar, d> &direction) const override {
            Vector<Scalar, d> corner;
            Scalar dd = direction.dot(direction);

            Scalar min = std::numeric_limits<Scalar>::infinity();
            Scalar max = -std::numeric_limits<Scalar>::infinity();

            for (int k = 0; k < (1 << d); ++k) {
                for (int i = 0; i < d; ++i)
                    corner[i] = bounds_[2 * i + ((k >> i) & 1)];
                Scalar v = direction.dot(corner) / dd;
                min = std::min(min, v);
                max = std::max(max, v);
            }

            return {min, max};
        }
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
            Scalar a = ray.direction.dot(ray.direction), b = 2 * o.dot(ray.direction), c =
                    o.dot(o) - radius * radius;
            Scalar D = b * b - 4 * a * c;
            if (D < 1e-12)
                return {};
            D = std::sqrt(D);
            a *= 2;
            return {{(-b - D) / a, (-b + D) / a}};
        }

        virtual std::pair<Scalar, Scalar> bounds(const Vector<Scalar, d> &direction) const override {
            Scalar dd = direction.dot(direction);
            Scalar c = direction.dot(center) / dd;
            Scalar r = radius / std::sqrt(dd);
            return {c - r, c + r};
        }
    };

}


#endif //STRANDS_DOMAIN_H
