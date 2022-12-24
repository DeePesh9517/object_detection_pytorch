# Object Detection with PyTorch pre-trained networks
### What are pre-trained networks?
* COCO dataset (Common Object in Context) tends to be the standard for object detection benchmarking.
* COCO dataset contains over 90 classes of common everyday objects.
* we will be using following SOTA classification networks
1. Faster R-CNN with ResNet50 (accurate but slow)
2. Faster R-CNN with MobileNet v3 backbone (fast but low accuracy)
3. RetinaNet with ResNet50 backbone (balanced)