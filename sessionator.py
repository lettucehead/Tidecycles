#!/usr/bin/env python
from pathlib import Path
from decimal import Decimal
from copy import copy
from sys import stdout
import csv

from sliding_window import window


SOURCE = '/Users/a_/pdm3h-hc6hj/tides/sonamun/_mandala/2023-2042-sixmin-data.csv'

DATETIME = 'Datetime'
PREDICTION = 'Prediction'
THRESHOLD = 0.3


def read():
    with open(SOURCE) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row[PREDICTION] = Decimal(row[PREDICTION])
            yield row


def write():
    writer = csv.writer(stdout)
    writer.writerow(('Start Datetime', 'End Datetime'))
    try:
        while True:
            start, end = (yield)
            writer.writerow((start[DATETIME], start[PREDICTION], end[DATETIME], end[PREDICTION]))
    except GeneratorExit:
        pass
    stdout.close()


def main():
    start = end = None
    sink = write()
    sink.send(None)
    for prev, cur, next in window(read(), 3):
        if (cur[PREDICTION] < THRESHOLD) and (prev[PREDICTION] >= THRESHOLD):
            start = copy(cur)
        if (cur[PREDICTION] < THRESHOLD) and (next[PREDICTION] >= THRESHOLD):
            end = copy(cur)
        if end is not None:
            sink.send((start, end))
            start = end = None


if __name__ == '__main__':
    main()
