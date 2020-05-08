# FCANet

> Junlong Cheng, Shengwei Tian, Long Yu, Hongchun Lu, Xiaoyi Lv, 2020. 

### Code organization
* `main.py: Model testing, including model loading, testing`
* `model/FCANet_res2net101: The model definition`
* `attention_block: Construction of spatial attention and channel attention modules`
* `our_loss.py: loss function,includes dice_loss、ce_dice_loss、jaccard_loss(IoU loss)、ce_jaccard_loss、tversky_loss`
* `metrics.py: precision、recall、accuracy、iou`

## Requirements
* Keras 2.2.4+
* tensorflow-gpu 1.9.0+
