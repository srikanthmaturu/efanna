#ifndef EFANNA_MATRIX_H
#define EFANNA_MATRIX_H

#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/array.hpp>


#include "distance.hpp"
namespace efanna {

template <typename T>
class Matrix {
public:
    Matrix(){}
    /**
     * Create a matrix using number of rows and cols with rawdata.
     * rawdata should be compact without any alignments.
     * Notice that efnn::Matrix itself just keep a reference of data and
     * does not make a copy of data, so keep it available.
     */
    Matrix(const size_t rows, const size_t cols, const void* data):
        rows_(rows), cols_(cols) {
#ifdef __GNUC__
#ifdef __AVX__
        align_cols = (cols + 7)/8*8;//re align to sse format
#else
#ifdef __SSE2__
        align_cols = (cols + 3)/4*4;
#else
        align_cols = cols;
#endif
#endif
#endif
        //std::cout<<" DD: "<<align_cols<<std::endl;
        for (size_t i = 0; i < rows; i++) {
            row_pointers_.push_back(reinterpret_cast<const T*>(data) + (align_cols * i));
            for(size_t j = 0; j < align_cols * i; j++){
                //std::cout << ((row_pointers_[row_pointers_.size() - 1]))[j] << "\t";
            }
            //std::cout <<  std::endl;
        }
    }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version){
        ar & rows_;
        ar & cols_;
        ar & align_cols;
        T* data = NULL;
        if (Archive::is_loading::value)
        {
            data = new T[rows_ * cols_];
        }
        else{
            data = const_cast<T*>(row_pointers_[0]);
        }

        ar & boost::serialization::make_array<T>(data, rows_ * cols_);

        if (Archive::is_loading::value)
        {
            for (size_t i = 0; i < rows_; i++) {
                row_pointers_.push_back(reinterpret_cast<const T*>(data) + (align_cols * i));
            }
        }
        std::cout << " Rows: " << rows_ << " Cols: " << cols_ << " Align Cols:" << align_cols <<" data sample:" << data[rows_ * cols_ - 2] << std::endl;
    }

    size_t get_cols() const {
        return cols_;
    }

    size_t get_rows() const {
        return rows_;
    }

    size_t get_align_cols() const {
        return align_cols;
    }

    const T* get_row(const size_t index) const {
        if (index >= rows_) {
            throw std::runtime_error("index out of range");
        }
        return row_pointers_[index];
    }

    // Debug usage only
    std::vector<std::pair<T, size_t> > brute_force_search(size_t idx, size_t k, Distance<T>* distance) const {
        printf("idx: %lu\n", idx);
        std::vector<std::pair<T, size_t> > result;
        for (size_t i = 0; i < rows_; i++) {
            result.push_back(std::make_pair(
                    distance->compare(get_row(i), get_row(idx), cols_),
                    i));
        }
        std::partial_sort(result.begin(), result.begin() + k, result.end());
        result.resize(k);
        return result;
    }
private:
    size_t rows_, cols_, align_cols;
    std::vector<const T*> row_pointers_;
};

}
#endif
