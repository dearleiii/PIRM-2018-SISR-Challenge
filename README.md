# PIRM-2018

## THE PIRM CHALLENGE ON PERCEPTUAL SUPER RESOLUTION

*PART OF THE PIRM WORKSHOP AT ECCV 2018* 

Single-image super-resolution has gained much attention in recent years. The appearance of deep neural-net based methods and the great advancement in generative modeling (e.g. GANs) has facilitated a major performance leap. One of the ultimate goals in super-resolution is to produce outputs with high visual quality, as perceived by human observers. However, many works have observed a fundamental disagreement between this recent leap in performance, as quantified by common evaluation metrics (PSNR, SSIM), and the subjective evaluation of human observers (reported e.g. in the SRGAN and EnhanceNet papers).

Reference:
[PIRM Challenge Webpage](https://www.pirm2018.org/PIRM-SR.html)



## Progress && Important dates

#### Prior work: 
  1. Generate 1600 HR image dataset 
    - [x] EnhancedNet bicubic & HR images - DIV2K800 
    - [x] EDSR HR images - DIV2K800
  2. F_perceptual score evaluation 
    - [x] EnhancedNet generated images
    - [x] EDSR generated HR images 
    - [x] Original HR images 
  3. Ma score, NIQE score, Perceptual score plots & evaluation 
    - [x] Ma score plot
    - [x] Perceptual score plot 
    - [x] NIQE score plot
  4. Approximator CNN-based Pytorch code -v1
    - [x] 2 Convolution layers, 1 FC layer, regular ReLU 
  
  
#### 07/13 Friday : 
#### 1. Build Approximator 
  - [x] Read & understand EDSR pytorch code structure
  - [x] Load dataset of size 2400
    - Combine sub-directory paths 
    - Solve Dataloader load '0.png' issue 
      - Current solution: add '0.png' file to edsr sub-directory
      - [ ] Todo: Re-write Data_loader class
  - [x] Imporve Batch_size ~ 100 batched 
    -  Based on prior knowledge that batch_size ~ 100 produce good training results 
  - [x] Analogy to the EDSR Discriminator code 
    - Set up 7 layers; -> adjust to 5 layers 
    - Set up LeadyReLU 
    - Check & Confirm structure is Differentiale 

  
  
#### 07/14 Saturday : 
  -  Check training result 
    - [x] Continuous score 
  - [x] Store training model
    - save_state_dict()

  - [ ] Store training model
    - [x] Store weights 
    - [ ] Store losses? 
    - [ ] Store final scatter plots
  - [x] chsh shell : /bin/bash 
    - modify ~/home/.profile file to set up env to be /bin/bash 
    - modify tmux.config file 


#### 07/15 Sunday : 
#### Prioity Run on GPU
  - Required memory : 12 ~ 50 GB
  
#### 1. Evaluate Approximator
  - [x] Scattor plot for training result vs. original results 
  - [ ] Link scatter plot results with MSE score 
  - [ ] Normalize MESloss() i.e./800

  
#### 2. Test Approximator
  - [x] Scatter plot for testing result vs. original testing dataset score 

#### 3. Try out different Approximator structure 



#### 07/16 Monday : 
#### Write DataParallel for multiple Gpus
  
#### Try out baseline model with small dataset size 


#### 07/17 Tuesday : 

#### [ ] 1.Experiment log file
#### [ ] 2.Training data parameter tunning 
#### [ ] 3.Save models 
#### [ ] 4. Training / testing loss plot
#### [ ] 5. Memory requirement analysis 

#### Combine with GAN structure 
  - Modify loss function to include both HR & LR datasets
  - Prepare both types of datasets

#### 07/18 Wednesday: 
#### Test data released 



#### 07/25 :
#### Submission Dealine 


#### 08/01: 
#### Challenge results released 
#### Code reconstruction 
#### 1. Re-write Dataloader()
  - Combine if include sub-directories
  - Solve the '0.png'
  - Load batches 

#### 08/22: 
#### Paper submission deadline 


## Reference Listing
1. [Intuitive explanation of CNN](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) 
2. [ ] Dehazing Paper - 2018 CVPR-NTIRE workshop 
3. [ ] New Techniques for Preserving Global Structure and Denoising in SISR - Duke Prediction Lab 2018 CVPR-NTIRE workshop
  - Consult authors about trickes that can be applied generally 
