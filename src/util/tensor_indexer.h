#ifndef SCHRODINGER2D_TENSOR_INDEXER_H
#define SCHRODINGER2D_TENSOR_INDEXER_H

#include <memory>
#include <Eigen/Core>
#include "per_direction.h"

namespace schrodinger {
    template<Eigen::Index dimension>
    class TensorIndexer {
        using Index = Eigen::Index;

        PerDirection<Index, dimension> m_sizes;
        PerDirection<Index, dimension> m_strides;
        Index m_totalSize;
    public:
        template<typename Indexable>
        explicit TensorIndexer(const Indexable &sizes_indexable) {
            Index stride = 1;
            m_totalSize = 1;
            for (Index i = 0; i < dimension; ++i) {
                m_sizes[i] = sizes_indexable[i];
                m_strides[i] = stride;
                stride *= m_sizes[i];
                m_totalSize *= m_sizes[i];
            }
        }

        [[nodiscard]] Index totalSize() const {
            return m_totalSize;
        }

        template<typename Indexable>
        Index operator()(const Indexable &index) const {
            Index r;
            for (Index i = 0; i < dimension; ++i)
                r += m_strides[i] * index[i];
            return r;
        }

        PerDirection<Index, dimension> inverse(Index index) const {
            PerDirection<Index, dimension> r;
            for (Index i = 0; i < dimension; ++i) {
                Index direction = index % m_strides[i];
                r[i] = direction;
                index -= direction;
            }
            return r;
        }

        template<typename Callback>
        void forEach(Callback cb) {
            for (Index i = 0; i < m_totalSize; ++i)
                cb(inverse(i));
        }

        template<typename Callback>
        void forEachDirectionStart(Index direction, Callback cb) {
            Index size = m_sizes[direction];
            Index stride = m_strides[direction];
            Index leftStep = m_totalSize / stride / size;
            for (Index leftOffset = 0; leftOffset < m_totalSize; leftOffset += leftStep)
                for (Index rightOffset = 0; rightOffset < stride; ++rightOffset)
                    cb(inverse(leftOffset + rightOffset), size);
        }
    };
}

#endif //SCHRODINGER2D_TENSOR_INDEXER_H
