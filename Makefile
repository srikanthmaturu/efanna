GXX=g++ -std=c++11
#OPTM=-O3 -msse2 -msse4 -fopenmp
OPTM=-O3 -march=native -fopenmp
CPFLAGS=$(OPTM) -Wall -DINFO
LDFLAGS=$(OPTM) -Wall -lboost_timer -lboost_chrono -lboost_system -DINFO

INCLUDES=-I./ -I./algorithm -I./general

SAMPLES=$(patsubst %.cc, %, $(wildcard samples/*.cc samples_hashing/*.cc))
SAMPLE_OBJS=$(foreach sample, $(SAMPLES), $(sample).o)

HEADERS=$(wildcard ./*.hpp ./*/*.hpp)

#EFNN is currently header only, so only samples will be compiled

#SHARED_LIB=libefnn.so
#OBJS=src/efnn.o

all: $(SHARED_LIB) $(SAMPLES)

#$(SHARED_LIB): $(OBJS)
#	$(GXX) $(LDFLAGS) $(LIBS) $(OBJS) -shared -o $(SHARED_LIB)

$(SAMPLES): %: %.o
	$(GXX) $^ -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp $(HEADERS)
	$(GXX) $(CPFLAGS) $(INCLUDES) -c $*.cpp -o $@

%.o: %.cc $(HEADERS)
	$(GXX) $(CPFLAGS) $(INCLUDES) -c $*.cc -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(SHARED_LIB)
	rm -rf $(SAMPLES)
	rm -rf $(SAMPLE_OBJS)

tf_idf_efanna_index:
	$(GXX) $(CPFLAGS) src/index.cpp -o tf_idf_efanna_index -I ./tf_idf_index  -I ./ -I ./algorithm -I ./general -pthread \
	-D DATA_TYPE=float -D NGRAM_LENGTH=3 -D USE_TDFS=false -D USE_IIDF=false -D TOTAL_NUMBER_OF_TREES=32 -D CONQUER_TO_DEPTH=8 -D ITERATION_NUMBER=8 \
	-D L_CD=200 -D CHECK=200 -D K_CD=100 -D S_CD=10 -D NUMBER_OF_TREES_FOR_BUILDING_GRAPH=8 $(LDFLAGS) -l boost_serialization -l boost_filesystem

