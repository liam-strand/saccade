CC = gcc
CFLAGS = -O2 -march=native -mtune=native
CONFIG_MODULE_SIG=n

obj-m += perf_bench_mod.o

all: perf_bench_mod perf_bench_user

perf_bench_mod:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f perf_bench_user
