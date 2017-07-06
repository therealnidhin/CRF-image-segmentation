# OBJ-Detection

We are using the object detection api provided by tensorflow to create the solution.

1. Install tensorflow and then clone tensorflow models libraries - https://github.com/tensorflow/models.
2. Follow installation steps documented in the models/object_detection.
3. Copy annotations and images folder to models/object_detection. Images contain the actual images while annotations contain xmls specifying bounding boxes for each image.
4. Copy create_fashion_tf_record.py to models/object_detection.
5. Copy fashion_label_map.pbtxt to models/object_detection/data
6. Execute command from object_detection folder "python create_fashion_tf_record.py     --label_map_path=data/fashion_label_map.pbtxt     --data_dir=`pwd`     --output_dir=`pwd`". 
This will create two files fash_train.record and fash_val.record. These files will be fed to the tensorflow network. 
7. Copy fash_train.record and fash_val.record to data folder.
8. Create folder called fash-model in models.
9. Copy faster_rcnn_resnet101_fash.config to fash-model.
10. Create folders named train and eval inside fash-model folder.
11. Download othe COCO-pretrained Faster R-CNN with Resnet-101 model. Unzip the contents of the folder and copy the model.ckpt* files into fash-model folder. (http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)
12. Start training by executing the folllowing command from object_detection folder - "python train.py     --logtostderr     --pipeline_config_path=models/fash-model/faster_rcnn_resnet101_fash.config  --train_dir=models/fash-model/train". 
Training goes on indefinitely till its killed by the user.
13. Execute "tensorboard --logdir=models/fash-model" to see and visualize the training and eval phases.
14. For evaluation execute the following command from object_detection folder - "python eval.py     --logtostderr     --pipeline_config_path=models/fash-model/faster_rcnn_resnet101_fash.config     --checkpoint_dir=models/fash-model/train     --eval_dir=models/fash-model/eval". 
This command will periodically fetch the latest checkoint from models/fash-model/train and perform evalutions. Take images tab in tensorboard ui to see evaluation results. 

Since the dataset is very small we can see some noise in the evalution results. Even then the correct catgories were detected in each image with the higest confidence.
