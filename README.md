# PIRM-2018

## THE PIRM CHALLENGE ON PERCEPTUAL SUPER RESOLUTION

*PART OF THE PIRM WORKSHOP AT ECCV 2018* 

Single-image super-resolution has gained much attention in recent years. The appearance of deep neural-net based methods and the great advancement in generative modeling (e.g. GANs) has facilitated a major performance leap. One of the ultimate goals in super-resolution is to produce outputs with high visual quality, as perceived by human observers. However, many works have observed a fundamental disagreement between this recent leap in performance, as quantified by common evaluation metrics (PSNR, SSIM), and the subjective evaluation of human observers (reported e.g. in the SRGAN and EnhanceNet papers).

Reference:
[PIRM Challenge Webpage](https://www.pirm2018.org/PIRM-SR.html)



## Progress && Important dates

#### Prior work: 
  1. Generate 1600 HR image dataset 
    - [ ] EnhancedNet bicubic & HR images - DIV2K800 
    - [ ] EDSR HR images - DIV2K800
  2. F_perceptual score evaluation 
    - [ ] EnhancedNet generated images
    - [ ] EDSR generated HR images 
    - [ ] Original HR images 
  3. Ma score, NIQE score, Perceptual score plots & evaluation 
    - [ ] Ma score plot
    - [ ] Perceptual score plot 
    - [ ] NIQE score plot
  
  
#### 07/13 Friday : 
#### 1. Build Approximator 
  - [ ] Imporve Batch_size ~ 100 batched 
    -  Based on prior knowledge that batch_size ~ 100 produce good training results 
  - [ ] Analogy to the EDSR Discriminator code 
    - Set up 7 layers;
    - Set up LeadyReLU 
    - Check & Confirm structure is Differentiale 
  - [ ] Check training result 
    - Continuous score 

#### 2. Evaluate Approximator
  - [ ] Scattor plot for training result vs. original results 
  - [ ] Link scatter plot results with MSE score 
  - [ ] Normalize MESloss() i.e./800
  
  
#### 07/14 Saturday : 
#### 1. Test Approximator
  - Scatter plot for testing result vs. original testing dataset score 

#### 2. Try out different Approximator structure 


#### 07/15 Sunday : 
#### 1. Combine with GAN structure 
  - Modify loss function to include both HR & LR datasets
  - Prepare both types of datasets
  - Start training 


#### 07/16 Monday : 


#### 07/17 Tuesday : 


#### 07/18 Wednesday: 
#### Test data released 


#### 08/01: 
#### Challenge results released 


#### 08/22: 
#### Paper submission deadline 


## Reference Listing
1. Dehazing Paper - 2018 CVPR-NTIRE workshop 
2. New Techniques for Preserving Global Structure and Denoising in SISR - Duke Prediction Lab 2018 CVPR-NTIRE workshop 
