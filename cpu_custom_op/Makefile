CC = g++

all: cpu_op.so

cpu_op.so: cpu_code.cpp
	$(CC) -fpic cpu_code.cpp -shared -lpoplar -Wl,-soname,$@ -o $@

clean:
	rm cpu_op.so
