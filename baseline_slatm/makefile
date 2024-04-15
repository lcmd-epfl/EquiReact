USE_OMP = 1

CC=gcc
OMPFLAG=
ifeq ($(USE_OMP),1)
	OMPFLAG = -fopenmp
endif

W= -Warray-bounds -Wmaybe-uninitialized -Wmissing-braces -Wparentheses -Wsequence-point -Wtype-limits -Wundef \
   -Wuninitialized -Wunused -Wmisleading-indentation -Wempty-body -Wunused-but-set-variable -Wunused-parameter -Winline

all: manh.so

%.so: %.o
	$(CC) $^ -shared -Wl,-soname,$@ -o $@ $(OMPFLAG)
%.o: %.c
	$(CC) $(OMPFLAG) $(W) $^ -c --std=gnu11 -O2 -fPIC -o $@

clean:
	rm -vf manh.so
