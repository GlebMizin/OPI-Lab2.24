#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from threading import Thread
from queue import Queue
from math import cos, pi, log, sin


eps = .0000001


def inf_sum(x, queue_1):
    summa = x
    prev = 0
    i = 1
    while abs((summa - prev) > eps):
        prev = summa
        summa += (cos(x*i))/i
        i += 1

    queue_1.put(summa)


def check(x, queue_1):

    summa = queue_1.get()
    result = -1 * log(2*sin(x/2))

    print(f"The sum is: {summa}")
    print(f"The check sum is: {result}")


if __name__ == '__main__':

    queue_1 = Queue()

    x = pi

    thread_1 = Thread(target=inf_sum(x, queue_1))
    thread_2 = Thread(target=check(x, queue_1))

    thread_1.start()
    thread_2.start()
