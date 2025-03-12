# RF Transmitter and Protocol Classification 
This repository presents a comprehensive approach to RF signal classification on IQ samples. By leveraging 1D CNNs and a multi-channel input strategy this approach achieves impressive accuracy on 
RF transsmitter classifiction and protocol classification. Models have been tested using a dataset taken at the University of Utah POWDER site and can found at the following link:https://repository.library.northeastern.edu/files/neu:gm80mp276.
On the givene dataset the model achieves 86% accuracy in protocol classification, 100% in transmitter classification, and 92% in joint classification tasks. The CNN’s also act as efficient modeling
solutions, as they can achieve also 100% performance in approx.10 seconds for classifying the transmitter and approx.20 seconds for classifying protocols and joint classification.

## Installation
    pip install -r requirements.txt
### How to run Training and Evaluation
    
    python train.py
        --data_dir "GlobecomPOWDER/"
        --task "transmitter"
        --save_dir "rf_results"
        --max_samples 20000
        --batch_sz 128
        --epochs 25
        --device "cuda"


