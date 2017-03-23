CC      :=  g++
CFLAGS  :=  -fPIC -Wall -Wextra -O2 -g
LDFLAGS :=  -shared
NAME    :=  scythe
DNAME   :=  $(NAME).lib # TODO : manage .so files
SRC     :=  *.cpp
OBJS    :=  $(SRC:%.cpp=%.o)
CFLAGS  :=  -ansi -pedantic -Wall -Werror -W -g3
LDFLAGS :=  -L.
LDLIBS  :=  -l$(...)

scythe.lib: scythe.o
	$(CXX) -shared $^ -o $@ -Wl,--whole-archive -Wl,--no-whole-archive