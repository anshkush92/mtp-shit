python train.py --trainRoot datasets/lmdb/IIIT5K/train/ \
--valRoot datasets/lmdb/IIIT5K/test/ --finetune false --cuda \
--arch crnn --lan english --charlist characters/english.txt \
--batchSize 32 --nepoch 15 --displayInterval 10 --valInterval 100 --adadelta --random_sample --deal_with_lossnan 
