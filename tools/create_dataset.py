import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import csv


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            # Encode key and value properly
            if isinstance(v, str):
                v = v.encode('utf-8')  # Encode string values
            txn.put(k.encode('utf-8'), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=10 * 1024 * 1024 * 1024)  # 10GB
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = f'image-{cnt:09d}'
        labelKey = f'label-{cnt:09d}'
        cache[imageKey] = imageBin  # Image data as bytes
        cache[labelKey] = label  # Label as string
        if lexiconList:
            lexiconKey = f'lexicon-{cnt:09d}'
            cache[lexiconKey] = ' '.join(lexiconList[i])  # Lexicon as string
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)  # Store total sample count
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def parse_args():
    parser = argparse.ArgumentParser(description="Create an LMDB dataset for CRNN training.")
    parser.add_argument('--outputPath', type=str, required=True, help="Path to save the LMDB dataset.")
    parser.add_argument('--csvFile', type=str, required=True, help="CSV file with ImgName, GroundTruth, and lexicons.")
    parser.add_argument('--lexicons', type=str, nargs='*', help="(Optional) List of lexicons for each image.")
    parser.add_argument('--checkValid', type=bool, default=True, help="Whether to check image validity (default: True).")
    parser.add_argument('--imagePrefix', type=str, default='datasets/IIIT5K/', help="Prefix to prepend to each ImgName.")
    
    return parser.parse_args()


def load_csv(csv_file, image_prefix='datasets/IIIT5K/'):
    """
    Load a CSV file and extract the image paths, ground truth labels, and lexicons.

    :param csv_file: Path to the CSV file.
    :param image_prefix: Prefix to prepend to each ImgName (default 'datasets/IIIT5K/').
    :return: imagePaths, labels, lexicons
    """
    imagePaths = []
    labels = []
    smallLexicons = []
    mediumLexicons = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)  # Use DictReader to access columns by name
        for row in reader:
            img_name = row['ImgName']
            ground_truth = row['GroundTruth']
            small_lexicon = row.get('smallLexi', '')
            medium_lexicon = row.get('mediumLexi', '')
            
            # Prepend the image prefix
            full_image_path = os.path.join(image_prefix, img_name)
            imagePaths.append(full_image_path)
            labels.append(ground_truth)
            smallLexicons.append(small_lexicon)
            mediumLexicons.append(medium_lexicon)

    # Return lexicons as well, if available
    return imagePaths, labels, smallLexicons, mediumLexicons


if __name__ == '__main__':
    # Parse arguments from the command line
    args = parse_args()

    # Load image paths, labels, and lexicons from CSV file
    imagePaths, labels, smallLexicons, mediumLexicons = load_csv(args.csvFile, image_prefix=args.imagePrefix)

    # Call createDataset function with the parsed arguments
    createDataset(args.outputPath, imagePaths, labels, lexiconList=smallLexicons, checkValid=args.checkValid)
    # Optionally, you could also pass mediumLexicons as lexiconList if you need that one
