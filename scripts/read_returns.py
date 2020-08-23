import csv
import os

dirname = 'HalfCheetah-v2'

def walk(dirname):
    for dirpath, dirname, filenames in os.walk(dirname):
        if 'progress.csv' in filenames:
            yield os.path.join(dirpath, 'progress.csv'), os.path.join(dirpath, 'variant.json')

def view_returns(progress):
    with open(progress, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 10 == 0:
                print(i, row['Test Returns Mean'])

if __name__ == '__main__':
    for progress, params in walk(dirname):
        view_returns(progress)

