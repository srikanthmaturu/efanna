//
// Created by srikanth on 8/25/17.
//

#pragma once

#include <efanna.hpp>
#include "general/matrix.hpp"
#include "tf_idf_efanna_idx_helper.hpp"

#include <cstdint>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <math.h>
#include <ctime>


#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/timer/timer.hpp>
#include <stdlib.h>

namespace tf_idf_efanna_index {
template<class data_type_t, uint64_t ngram_length_t, bool use_tdfs_t, bool use_iidf_t, uint64_t total_number_of_trees_t, uint64_t conquer_to_depth_t, uint64_t iteration_number_t,
 uint64_t l_t, uint64_t check_t, uint64_t k_t, uint64_t s_t, uint64_t number_of_trees_for_building_graph_t>
    class tf_idf_efanna_idx {
    public:
        tf_idf_efanna_idx() = default;

        tf_idf_efanna_idx(const tf_idf_efanna_idx &) = default;

        tf_idf_efanna_idx(tf_idf_efanna_idx &&) = default;

        tf_idf_efanna_idx &operator=(const tf_idf_efanna_idx &) = default;

        tf_idf_efanna_idx &operator=(tf_idf_efanna_idx &&) = default;

        tf_idf_efanna_idx(std::vector<std::string> &data) {

        }

        typedef data_type_t data_type;
        typedef efanna::Matrix<data_type> Dataset;

        efanna::FIndex<float> * index;
        Dataset dataset;

        bool use_tdfs = use_tdfs_t;
        bool use_iidf = use_iidf_t;

        std::vector<std::string> original_data;
        uint64_t ngram_length = ngram_length_t;
        std::vector<uint64_t> tdfs;
        std::vector<data_type> center;

        uint64_t total_number_of_trees = total_number_of_trees_t;
        uint64_t conquer_to_depth = conquer_to_depth_t;
        uint64_t iteration_number = iteration_number_t;
        uint64_t l = l_t;
        uint64_t check = check_t;
        uint64_t k = k_t;
        uint64_t s = s_t;
        uint64_t number_of_trees_for_building_graph = number_of_trees_for_building_graph_t;

        std::map<char, int> a_map = {{'A', 0},
                                     {'C', 1},
                                     {'G', 2},
                                     {'T', 3},
                                     {'N', 4}};

        void store_to_file(std::string data_file) {
            std::ofstream idx_file_ofs(data_file);
            boost::archive::binary_oarchive oa(idx_file_ofs);
            oa << original_data;
            oa << tdfs;
            oa << dataset;
            oa << center;
        }

        void load_from_file(std::string data_file) {
            original_data.clear();
            tdfs.clear();
            std::ifstream idx_file_ifs(data_file);
            boost::archive::binary_iarchive ia(idx_file_ifs);
            ia >> original_data;
            ia >> tdfs;
            ia >> dataset;
            ia >> center;
        }

        std::vector< std::vector< pair<std::string, uint64_t > > > match(std::vector<std::string>::iterator queries_begin_iterator, std::vector<std::string>::iterator queries_end_iterator, uint64_t number_of_queries) {
            std::vector< std::vector< pair<std::string, uint64_t > > > query_results_vector;
            uint64_t n_cols = dataset.get_cols();
            data_type * query_data = new data_type[number_of_queries * n_cols];
            uint64_t i = 0;
            for(std::vector<std::string>::iterator it = queries_begin_iterator; it != queries_end_iterator; it++, i++){
                std::vector<data_type> query_tf_idf_vector = getQuery_tf_idf_vector(*it);
                for(uint64_t j = 0; j < query_tf_idf_vector.size(); j++){
                    query_data[i * n_cols + j] = query_tf_idf_vector[j];
                }
            }

            int search_trees = total_number_of_trees;
            int search_epoc = 4;
            int poolsz = 1200;
            int search_extend = 200;
            int search_method = 0;
            int kNN = 20;
            index->setSearchParams(search_epoc, poolsz, search_extend, search_trees, search_method);

            efanna::Matrix<data_type> query_matrix(queries.size(), n_cols, query_data);
            boost::timer::auto_cpu_timer timer;
            index->knnSearch(kNN, query_matrix);
            std::cout<<timer.elapsed().wall / 1e9<<std::endl;

            std::cout << std::endl;
            for(uint64_t i = 0; i < queries.size(); i++){
                std::vector<int32_t> nearestNeighbours;
                uint8_t minED = 100;
                for(size_t k=0; k < nearestNeighbours.size(); ++k){
                    uint64_t edit_distance = uiLevenshteinDistance(queries[i], original_data[nearestNeighbours[k]]);
                    if(edit_distance == 0){
                        continue;
                    }
                    if(edit_distance < minED){
                        minED = edit_distance;
                        query_results_vector[i].clear();
                    }
                    else if(edit_distance > minED){
                        continue;
                    }
                    query_results_vector[i].push_back(make_pair(original_data[nearestNeighbours[k]], edit_distance));
                }
                std::cout << "Processed query: " << i << std::endl;
            }
            return query_results_vector;
        }

        void initialize_index(){
            index = new efanna::FIndex<float>(dataset, new efanna::L2DistanceAVX<data_type>(),
                                  efanna::KDTreeUbIndexParams(true, total_number_of_trees, conquer_to_depth, iteration_number, check, l, k, total_number_of_trees, s));
        }

        void build_index(){
            clock_t s,f;
            s = clock();
            index->buildIndex();
            f = clock();
            cout<<"Index building time : "<<(f-s)*1.0/CLOCKS_PER_SEC<<" seconds"<<endl;
        }

        void load_index(std::string idx_file){
            index->loadIndex(idx_file);
        }

        void store_index(std::string idx_file){
            index->saveIndex(idx_file);
        }

    std::vector<data_type> getQuery_tf_idf_vector(std::string query) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t string_size = query.size();
            uint64_t data_size = dataset.get_rows();
            std::vector<data_type> tf_idf_vector(tf_vec_size, 0);
            for (uint64_t i = 0; i < string_size - ngram_length + 1; i++) {
                std::string ngram = query.substr(i, ngram_length);
                uint64_t d_num = 0;
                for (uint64_t j = 0; j < ngram_length; j++) {
                    d_num += a_map[ngram[j]] * pow(4, (ngram_length - j - 1));
                }
                tf_idf_vector[d_num]++;
            }
            double vec_sq_sum = 0.0;
            for (uint64_t i = 0; i < tf_vec_size; i++) {
                if (tf_idf_vector[i] > 0) {
                    if (!use_tdfs) {
                        tf_idf_vector[i] = (1 + log10(tf_idf_vector[i]));
                    } else if (!use_iidf) {
                        if (tdfs[i] > 0) {
                            tf_idf_vector[i] *= ((log10(1 + (data_size / tdfs[i]))));
                        }
                    } else {
                        tf_idf_vector[i] *= ((log10(1 + ((double) tdfs[i] / (double) data_size))));
                    }
                    vec_sq_sum += pow(tf_idf_vector[i], 2);
                }
            }
            vec_sq_sum = pow(vec_sq_sum, 0.5);

            for (uint64_t i = 0; i < tf_vec_size; i++) {
                tf_idf_vector[i] /= vec_sq_sum;
                //std::cout << tf_idf_vector[i] << " \t";
            }
            //std::cout << std::endl;
            subtract_center(tf_idf_vector);
            return tf_idf_vector;
        }

        void construct_dataset(std::vector<std::string> &data) {
            uint64_t tf_vec_size = pow(4, ngram_length);
            uint64_t data_size = data.size();
            uint64_t string_size = data[0].size();
            tdfs.resize(tf_vec_size);
            std::vector<double> vec_sq_sums(data_size);

            data_type* tf_idf_vectors = (data_type*)memalign(KGRAPH_MATRIX_ALIGN, data_size * tf_vec_size * sizeof(data_type));

            for (uint64_t i = 0; i < data_size; i++) {
                for (uint64_t j = 0; j < string_size - ngram_length + 1; j++) {
                    std::string ngram = data[i].substr(j, ngram_length);
                    uint64_t d_num = 0;
                    for (uint64_t k = 0; k < ngram_length; k++) {
                        d_num += a_map[ngram[k]] * pow(4, (ngram_length - k - 1));
                    }
                    tf_idf_vectors[i][d_num]++;
                }
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (tf_idf_vectors[i][j] > 0) {
                        tf_idf_vectors[i][j] = (1 + log10(tf_idf_vectors[i][j]));
                        if (use_tdfs) {
                            tdfs[j]++;
                        }
                    }
                }
            }
            for (uint64_t i = 0; i < data_size; i++) {
                double_t vec_sq_sum = 0.0;
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    if (use_tdfs) {
                        if (!use_iidf) {
                            if (tdfs[j] > 0) {
                                tf_idf_vectors[i][j] *= (log10(1 + (data_size / tdfs[j])));
                            }
                        } else {
                            tf_idf_vectors[i][j] *= (log10(1 + ((double) tdfs[j] / (double) data_size)));
                        }
                    }
                    vec_sq_sum += pow(tf_idf_vectors[i][j], 2);
                }
                vec_sq_sum = pow(vec_sq_sum, 0.5);
                for (uint64_t j = 0; j < tf_vec_size; j++) {
                    tf_idf_vectors[i][j] /= vec_sq_sum;
                    //std::cout << tf_idf_vectors[i][j] << " \t";
                }
                //std::cout << std::endl;
            }

            dataset = Dataset(data_size, tf_vec_size, data);
            re_center_dataset();
        }

        void re_center_dataset() {
            // find the center of mass
            uint64_t data_size = dataset.get_rows();
            data_type * row;
            for(uint64_t i = 0; i < dataset.get_rows(); i++){
                row = dataset.get_row[i];
                for(uint64_t j = 0; j < dataset.get_cols(); j++){
                    center[j] = row[j];
                }
            }

            for(uint64_t i = 0; i < dataset.get_rows(); i++){
                center[j] /= data_size;
            }

            std::cout << "Re-centering dataset points." << std::endl;
            for(uint64_t i = 0; i < dataset.get_rows(); i++){
                row = dataset.get_row[i];
                for(uint64_t j = 0; j < dataset.get_cols(); j++){
                    row[j] -= center[j];
                }
            }
            std::cout << "Done." << std::endl;
        }

        void subtract_center(vector<data_type>& point) {
            for(uint64_t i = 0; i < point.size(); i++){
                point[i] -= center[i];
            }
        }
    };
}
