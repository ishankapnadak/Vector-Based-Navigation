Tried training the network on distance as well as sine and cosine of the head direction angle. Moreover, I increase the weight of the sine and cosine component in the overall loss by a factor of 10. The following loss curves were obtained:

**Norm Loss**:

![Norm Loss](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_angle_10/losses/norm_loss.jpg)

**Sine Loss**:

![Sine Loss](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_angle_10/losses/sine_loss.jpg)


**Cosine Loss**:

![Cosine Loss](https://github.com/ishankapnadak/Vector-Based-Navigation/blob/main/New%20Supervised/distance_angle_10/losses/cosine_loss.jpg)

The R<sup>2</sup> coefficient for norm, sine, and cosine are as follows: 
1. 0.657
2. -61.014
3. -0.12
  
Conclusion: The network was not able to learn the sine and cosine at all, and its performance on the distance also deteriorated.
