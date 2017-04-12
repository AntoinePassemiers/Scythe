CC           :=  g++
FLAGS        :=  -std=c++14 -Iinclude
CFLAGS       :=  -ggdb3 -Wpedantic -Wall -Wextra -Wconversion -Weffc++ -Wstrict-null-sentinel -Wold-style-cast -Wnoexcept -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wzero-as-null-pointer-constant -Wconversion -pthread
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

OBJS := cart.o scythe.o classification_forest.o regression_forest.o
OBJS += classification_metrics.o regression_metrics.o

$(TARGET): $(OBJS)
	$(CC) $(FLAGS) $(LDFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJS)
	@rm -rf $(OBJS) # clean

regression_metrics.o: metrics/regression_metrics.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_metrics.o: metrics/classification_metrics.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

regression_forest.o: forest/regression_forest.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

classification_forest.o: forest/classification_forest.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

%.o: %.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<