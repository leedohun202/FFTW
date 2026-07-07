CC = gcc
CFLAGS = -O3 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math -ftree-vectorize -fopenmp -Wall \
         -I./Globals -I./FFT -I./Pipeline -I./Utils
LDFLAGS = -lfftw3f -lfftw3f_threads -lpthread -lm
TARGET = radar_test

SRCS = main.c \
       $(wildcard Globals/*.c) \
       $(wildcard FFT/*.c) \
       $(wildcard Pipeline/*.c) \
       $(wildcard Utils/*.c)
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)