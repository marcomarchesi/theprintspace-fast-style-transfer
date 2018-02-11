from binascii import a2b_base64
from argparse import ArgumentParser
import os
import csv
import base64
import sys
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)

parser = ArgumentParser()
parser.add_argument('--tsv_path', dest='tsv_path')
parser.add_argument('--out_path', dest='out_path')
args = parser.parse_args()

# load image data from TSV
def create_png(path, name, data):
    binary_data = a2b_base64(data)
    encoded_data = base64.b64encode(binary_data)
    filename = name + '.jpg'
    # write png file
    with open(os.path.join(path, filename), 'wb') as fd:
        fd.write(base64.decodebytes(encoded_data))
        fd.close()

with open(args.tsv_path) as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for i, row in enumerate(tqdm(reader)):
        content = row[5]
        create_png(args.out_path, str(i), content)
        if i >= 80000:
            break




