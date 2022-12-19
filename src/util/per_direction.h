#ifndef SCHRODINGER2D_PER_DIRECTION_H
#define SCHRODINGER2D_PER_DIRECTION_H

#include <array>
#include <cassert>
#include <variant>

namespace schrodinger {

    template<typename PD, typename Type, int dimension>
    struct PerDirectionBase {
    private:
        PD &derived() {
            return *static_cast<PD *>(this);
        }

        const PD &derived() const {
            return *static_cast<const PD *>(this);
        }

    public:

        template<int i>
        Type &get() {
            static_assert(0 <= i && i < dimension);
            return derived().data()[i];
        }

        template<int i>
        const Type &get() const {
            static_assert(0 <= i && i < dimension);
            return derived().data()[i];
        }

        Type &operator[](int i) {
            assert(0 <= i && i < PD::dimension);
            return derived().data()[i];
        }

        const Type &operator[](int i) const {
            assert(0 <= i && i < PD::dimension);
            return derived().data()[i];
        }

        operator Eigen::Matrix<Type, dimension, 0>() const {
            return Eigen::Map<Eigen::Matrix<Type, dimension, 0>>(derived().data());
        }
    };

    template<typename T, int dimension>
    struct PerDirection : public PerDirectionBase<PerDirection<T, dimension>, T, dimension> {
    private:
        std::array<T, dimension> m_data;
    public:
        const T *data() const {
            return m_data.data();
        }

        T *data() {
            return m_data.data();
        }
    };

    template<typename T>
    struct PerDirection<T, 2> : public PerDirectionBase<PerDirection<T, 2>, T, 2> {
        static constexpr int dimension = 2;

        T x;
        T y;

        const T *data() const {
            return &x;
        }

        T *data() {
            return &x;
        }
    };
}

#endif //SCHRODINGER2D_PER_DIRECTION_H
