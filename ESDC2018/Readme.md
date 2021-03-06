# ID-Aware Facial Expression Detection and Analysis System
This is a group project done by me as well as other two schoolmate. It won the third prize in the ESDC 2018 worldwild. It combined codes for several already published papers and made some modifications so the design can detect and record people's emotion after recogizing their facial expression while at the same time tell different people apart. It is not a research project.
<br/><br/>
The design is based on Intel-Up2 board and a video camera, but you can also run it on your computer. It worked well on Linux and Mac. We have not tested it on windows. To run the codes, you need to install python2, pytorch, pip, OpenCV2, tensorflow, keras etc.. You also need to download a package named [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) from github. You can either put your own video clips into './Emotion_Detect/Emotion/demos/' or get interactive reponse using a camera.
<br/><br/>
For more details, please refer to the following reports:<br/>
[Identity-Aware Automatic Emotion Detection and Analysis System.pdf](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/Identity-Aware%20Automatic%20Emotion%20Detection%20and%20Analysis%20System.pdf)<br/>
[多对象情绪自动识别分析系统.pdf](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/%E5%A4%9A%E5%AF%B9%E8%B1%A1%E6%83%85%E7%BB%AA%E8%87%AA%E5%8A%A8%E8%AF%86%E5%88%AB%E5%88%86%E6%9E%90%E7%B3%BB%E7%BB%9F.pdf)
<br/><br/>
Copyrite Xinchen Wang, Xiaowen You, Jiakai Zhang, Chuang Liu, 2018.
## Demo
![Demo](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/Demo1.gif)
![Demo](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/RealTimeDemo.gif)

## System Design & Operation Flow
![alt text](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/img1.PNG)

## Emotional Change of the Trueman Clip
![alt text](https://github.com/arora123you/Xiaowen-s-personal-site/blob/master/ESDC2018/img2.PNG)

## Algorithm
CNN with Xception method and Global Polling Layer (GAP) to shrink training arguments. It not only reduced calculation complexity but also increased accuracy because redundant parameters were thrown away[4].
## References
[1] Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift[J]. 2015:448-456.
<br/>
[2]	Lin M, Chen Q, Yan S. Network in network[J]. arXiv preprint arXiv:1312.4400, 2013.
<br/>
[3]	Kingma D, Ba J. Adam: A Method for Stochastic Optimization[J]. Computer Science, 2014.
<br/>
[4]	Octavio Arriaga, Paul G. Plöger, Matias Valdenegro. Real-time Convolutional Neural Networks for Emotion and Gender Classification[J/OL], 2017,
https://github.com/oarriaga/face_classification/blob/master/report.pdf
