CC           :=  g++
FLAGS        :=  -std=c++14 -Iinclude
CFLAGS       :=  -ggdb3 -Wpedantic -Wall -Wextra -Wconversion -Weffc++ -Wstrict-null-sentinel -Wold-style-cast -Wnoexcept -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wzero-as-null-pointer-constant -Wconversion -pthread
CPPFLAGS     :=  -O2 -g
LDFLAGS      :=  -L. -shared
DEBUGFLAGS   :=  -O0 -D _DEBUG
RELEASEFLAGS :=  -O2 -D NDEBUG -combine -fwhole-program
NAME         :=  scythe
TARGET       :=  $(NAME).lib # TODO : manage .so files
SRC          :=  $(wildcard *.cpp)
OBJS         :=  $(SRC:.cpp=.o)


%.o: %.cpp
	$(CC) $(FLAGS) $(CFLAGS) $(CPPFLAGS) -c $<

$(TARGET): $(OBJS)
	$(CC) $(FLAGS) $(LDFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJS)
	@rm -rf $(OBJS) # clean
