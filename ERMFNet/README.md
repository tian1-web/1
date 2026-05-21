
# **ERMFNet: Evidence Reasoning Multimodal Fusion Network for Reliable Driving Behavior Recognition**


## 一、Dataset 




1.Origin UAH-DriveSet : The UAH-DriveSet is available at: http://www.robesafe.com/personal/eduardo.romera/uah-driveset.

2.Origin CL-Drive dataset: The CL-Drive dataset is available at: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JJ2YZZ.




## **二、Quick Start**

#### 1.Environment configuration

```
git clone https://github.com/1/ERFMNet.git
pip install -r requirements.txt  # install
```

#### 2.train

```
python main.py --mode train
```

#### 3.test

```
python main.py --mode test
```


# **Acknowledgements**

This code base is built upon and inspired by the following excellent works. We sincerely thank the authors for open-sourcing their code:

- **GLMDriveNet:** Liu W, Gong Y, Zhang G, et al. GLMDriveNet: Global–local multimodal fusion driving behavior classification network[J]. Engineering Applications of Artificial Intelligence, 2024, 129: 107575.