CC           :=  g++
FLAGS        :=  -std=c++14 -Iinclude
CFLAGS       :=  -ggdb3 -Wpedantic -Wall -Wextra -Wconversion -Weffc++ -Wstrict-null-sentinel -Wold-style-cast -Wnoexcept -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wzero-as-null-pointer-constant -Wconversion -pthread -O3
CPPFLAGS     :=  -O2 -g
LDFLAGS      :=  -L. -shared
DEBUGFLAGS   :=  -O0 -D _DEBUG
RELEASEFLAGS :=  -O2 -D NDEBUG -combine -fwhole-program
NAME         :=  scythe

ifeq ($(OS),Windows_NT)
	TARGET = $(NAME).lib
else
	TARGET = $(NAME).so
endif

OBJS := deep_scythe.o scythe.o cart.o classification_gb.o regression_gb.o
OBJS += classification_metrics.o regression_metrics.o
OBJS += classification_rf.o regression_rf.o
OBJS += classification_complete_rf.o regression_complete_rf.o
OBJS += bagging.o heuristics.o cascade.o scanner.o
OBJS += sets.o

$(TARGET): $(OBJS)
	$(CC) $(FLAGS) $(CFLAGS) $(LDFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJS)
	@rm -rf $(OBJS) # clean

cart.o : tree/cart.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

heuristics.o : tree/heuristics.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

sets.o : misc/sets.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

bagging.o : misc/bagging.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

regression_metrics.o: metrics/regression_metrics.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

regression_gb.o: forest/regression_gb.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

regression_rf.o: forest/regression_rf.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

regression_complete_rf.o: forest/regression_complete_rf.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_metrics.o: metrics/classification_metrics.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_gb.o: forest/classification_gb.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_rf.o: forest/classification_rf.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_complete_rf.o: forest/classification_complete_rf.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

cascade.o: deep_learning/cascade.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

scanner.o: deep_learning/scanner.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

%.o: %.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<