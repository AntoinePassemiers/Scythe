CC      :=  g++
CFLAGS  :=  -fPIC -Wall -Wextra -O2 -g -std=c++14 -ggdb3 -Wpedantic -Wall -Wextra -Wconversion -Weffc++ -Wstrict-null-sentinel -Wold-style-cast -Wnoexcept -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wzero-as-null-pointer-constant -Wsuggest-final-types -Wsuggest-final-methods -Wsuggest-overridec -pthread
LDFLAGS :=  -shared
NAME    :=  scythe
DNAME   :=  $(NAME).lib # TODO : manage .so files
SRC     :=  *.cpp
OBJS    :=  $(SRC:%.cpp=%.o)
CFLAGS  :=  -ansi -pedantic -Wall -Werror -W -g3
LDFLAGS :=  -L.
LDLIBS  :=  -l$(...)

scythe.lib: scythe.o id3.o metrics.o forest.o
	$(CXX) -shared $^ -o $@ $(CFLAGS)