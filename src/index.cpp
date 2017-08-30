//
// Created by srikanth on 8/25/17.
//

//
// Created by Srikanth Maturu (srikanthmaturu@outlook.com)on 7/17/2017.
//

#include "tf_idf_efanna_idx.hpp"

#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <omp.h>

using namespace std;
using namespace tf_idf_efanna_index;

using namespace std::chrono;
using timer = std::chrono::high_resolution_clock;

#ifdef CUSTOM_BOOST_ENABLED
const string index_name = INDEX_NAME;
#else
const string index_name = "TF_IDF_EFANNA_IDX";
#endif

string pt_name = "";

#define getindextype(ngl, utd, uiidf, th, pt) tf_idf_falconn_idx<ngl,utd,uiidf,th,pt>
#define STR(x)    #x
#define STRING(x) STR(x)

template<class duration_type=std::chrono::seconds>
struct my_timer{
    string phase;
    time_point<timer> start;

    my_timer() = delete;
    my_timer(string _phase) : phase(_phase) {
        std::cout << "Start phase ``" << phase << "''" << std::endl;
        start = timer::now();
    };
    ~my_timer(){
        auto stop = timer::now();
        std::cout << "End phase ``" << phase << "'' ";
        std::cout << " ("<< duration_cast<duration_type>(stop-start).count() << " ";
        std::cout << " ";
        std::cout << " elapsed)" << std::endl;
    }
};

template<class data_type_t, uint64_t ngram_length_t, bool use_tdfs_t, bool use_iidf_t, uint64_t total_number_of_trees_t, uint64_t conquer_to_depth_t, uint64_t iteration_number_t,
        uint64_t l_t, uint64_t check_t, uint64_t k_t, uint64_t s_t, uint64_t number_of_trees_for_building_graph_t>
struct idx_file_trait{
    static std::string value(std::string file_name){
        return file_name + ".NGL_" + to_string(ngram_length_t)+ "_UTD_" + ((use_tdfs_t)?"true":"false") + "_UIIDF_" + ((use_iidf_t)?"true":"false") +
                "_TNT_" +to_string(total_number_of_trees_t)+"_CTD_"+to_string(conquer_to_depth_t)+"_IN_"+to_string(iteration_number_t)+
                "_L_"+to_string(l_t)+"_CK_"+to_string(check_t)+"_k_"+to_string(k_t)+"_s_"+to_string(s_t)+"_NTBG_"+to_string(number_of_trees_for_building_graph_t);
    }
};

void load_sequences(string sequences_file, vector<string>& sequences){
    ifstream input_file(sequences_file, ifstream::in);

    for(string sequence; getline(input_file, sequence);){
        uint64_t pos;
        if((pos=sequence.find('\n')) != string::npos){
            sequence.erase(pos);
        }
        if((pos=sequence.find('\r')) != string::npos){
            sequence.erase(pos);
        }
        sequences.push_back(sequence);
    }
}

int main(int argc, char* argv[]){
    constexpr uint64_t ngram_length = NGRAM_LENGTH;
    typedef DATA_TYPE data_type;
    constexpr bool use_tdfs = USE_TDFS;
    constexpr bool use_iidf = USE_IIDF;
    constexpr uint64_t total_number_of_trees = TOTAL_NUMBER_OF_TREES;
    constexpr uint64_t conquer_to_depth = CONQUER_TO_DEPTH;
    constexpr uint64_t iteration_number = ITERATION_NUMBER;
    constexpr uint64_t l = L_CD;
    constexpr uint64_t check = CHECK;
    constexpr uint64_t k = K_CD;
    constexpr uint64_t s = S_CD;
    constexpr uint64_t number_of_trees_for_building_graph = NUMBER_OF_TREES_FOR_BUILDING_GRAPH;

    typedef INDEX_TYPE tf_idf_efanna_index_type;

    if ( argc < 3 ) {
        cout << "Usage: ./" << argv[0] << " sequences_file query_file" << endl;
        return 1;
    }

    if(use_tdfs)
    {
        cout << "Usage of tdfs is enabled." << endl;
    }
    else{
        cout << "Usage of tdfs is disabled." << endl;
    }

    string sequences_file = argv[1];
    string queries_file = argv[2];

    cout << "SF: " << sequences_file << " QF:" << queries_file << endl;
    string idx_file = idx_file_trait<data_type , ngram_length, use_tdfs, use_iidf, total_number_of_trees, conquer_to_depth, iteration_number,
            l, check, k, s, number_of_trees_for_building_graph>::value(sequences_file);
    string data_file = sequences_file + "_data_file";
    string queries_results_file = idx_file_trait<data_type , ngram_length, use_tdfs, use_iidf, total_number_of_trees, conquer_to_depth, iteration_number,
            l, check, k, s, number_of_trees_for_building_graph>::value(queries_file) + "_search_results.txt";
    tf_idf_efanna_index_type tf_idf_efanna_i;

    {
        ifstream idx_graphs_ifs(idx_file + ".graphs"), idx_trees_ifs(idx_file + ".trees");
        if ( !idx_graphs_ifs.good() || !idx_trees_ifs.good()){
            auto index_construction_begin_time = timer::now();
            vector<string> sequences;
            load_sequences(sequences_file, sequences);
            {
                cout<< "Index construction begins"<< endl;
                auto temp = tf_idf_efanna_index_type(sequences);
                tf_idf_efanna_i = std::move(temp);
            }

            tf_idf_efanna_i.store_to_file(data_file);
            tf_idf_efanna_i.initialize_index();
            tf_idf_efanna_i.build_index();
            tf_idf_efanna_i.store_index(const_cast<char*>(idx_file.c_str()));
            auto index_construction_end_time = timer::now();
            cout<< "Index construction completed." << endl;
            cout << "# total_time_to_construct_index_in_us :- " << duration_cast<chrono::microseconds>(index_construction_end_time-index_construction_begin_time).count() << endl;
        } else {
            cout << "Index already exists. Using the existing index." << endl;
            tf_idf_efanna_i.load_from_file(data_file);
            tf_idf_efanna_i.initialize_index();
            tf_idf_efanna_i.load_index(const_cast<char*>(idx_file.c_str()));
            std::cout << "Loaded from file. " << std::endl;
        }

        vector<string> queries;
        load_sequences(queries_file, queries);
        ofstream results_file(queries_results_file), linear_test_results_file("linear_test_results");
        cout << "Filter enabled. Filtering based on edit-distance. Only kmers with least edit-distance to query is outputted." << endl;
        auto start = timer::now();

        uint64_t block_size = 100000;
        uint64_t queries_size = queries.size();
        std::cout << queries_size << std::endl;
        if(queries_size < block_size){
            block_size = queries_size;
        }

        uint64_t extra_block = queries_size % block_size;
        uint64_t number_of_blocks =  queries_size / block_size;

        if(extra_block > 0) {
            number_of_blocks++;
        }

        for(uint64_t bi = 0; bi < number_of_blocks; bi++){
            uint64_t block_end = (bi == (number_of_blocks-1))? queries_size : (bi + 1)*block_size;
            auto query_results_vector = tf_idf_efanna_i.match(queries.begin()+bi * block_size, queries.begin()+block_end, (block_end - bi * block_size), bi * block_size);
            //#pragma omp parallel for
            for(uint64_t i= bi * block_size, j = 0; i< block_end; i++, j++){
                results_file << ">" << queries[i] << endl;
                //cout << "Stored results of " << i << endl;
                for(size_t k=0; k<query_results_vector[j].size(); k++){
                    results_file << "" << query_results_vector[j][k].first.c_str() << "  " << query_results_vector[j][k].second << endl;
                }
                query_results_vector.clear();
                cout << "Processed query: " << i << endl;
            }
        }

        auto stop = timer::now();
        cout << "# time_per_search_query_in_us = " << duration_cast<chrono::microseconds>(stop-start).count()/(double)queries.size() << endl;
        cout << "# total_time_for_entire_queries_in_us = " << duration_cast<chrono::microseconds>(stop-start).count() << endl;
        cout << "Saved results in the results file: " << queries_results_file << endl;
        results_file.close();
    }
}


