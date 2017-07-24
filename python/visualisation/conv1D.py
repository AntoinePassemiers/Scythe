import random

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk


class Iterator:
    def __init__(self, N, M, kc):
        self.N = N
        self.M = M
        self.kc = kc
        self.sc = self.M - self.kc + 1

    def _iterator_begin(self, j):
        self.i = 0
        self.x = j

    def _iterator_inc(self):
        self.i += 1
        if (self.i == self.sc):
            self.q += self.M
            self.i = 0
            if (self.q == self.sr * self.M):
                self.q = 0
                self.x += self.M * self.P

    def _iterator_deref(self):
        return self.x + self.i


class Grid:
    def __init__(self):
        self.master = tk.Tk()

        self.squares = list()
        pw, k = 20, 0
        self.cv = tk.Canvas(self.master, width = 850, height = 400)
        for i in range(10):
            for j in range(20):
                self.squares.append(
                    self.cv.create_rectangle(
                        j*pw, i*pw + k, (j+1)*pw, (i+1)*pw + k))
            k += 10

        self.iterator = Iterator(10, 20, 10)
        self.iterator._iterator_begin(47)
        self.last_id = None

        self.cv.bind("<i>", self.incrementStep)
        self.cv.pack()
        self.cv.focus_set()
        self.master.mainloop()

    def incrementStep(self, event = None):
        i = self.iterator._iterator_deref()
        if self.last_id is not None:
            self.cv.itemconfig(self.squares[self.last_id], fill = "blue")
        try:
            self.cv.itemconfig(self.squares[i], fill = "yellow")
        except IndexError:
            print("Out of bounds : %i" % i)
            return

        self.last_id = i
        self.iterator._iterator_inc()

if __name__ == "__main__":
    grid = Grid()
