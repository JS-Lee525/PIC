# Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation (ECCV 2024 Poster) 

*Sorry for working in progress! We should add some details for our code.

This is the official code of the paper "Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation" in ECCV 2024. 

![thumbnail](assets/thumbnail.png)

## News

:star: [2024. July] Our paper is accepted in ECCV 2024! \
:star: [2024. Sep] We've uploaded our video & poster for ECCV 2024! You can check them through this [link](https://eccv.ecva.net/virtual/2024/poster/2134). Also, the official code of our paper has been released! We are still updating the code for better performance.


## Getting Started

### Installing

```
git clone https://github.com/JS-Lee525/PIC.git
```

```
conda create -n [your_env] python=3.9
pip install -r requirements.txt
```

### Structures

    .
    ├── assets    
          ├── thumbnail.png      
    ├── sample_data
          ├── cat
                ├── cat_1.png
          ├── dog
                ├── dog_1.png
          ├── horse
                ├── horse_1.png
          ├── tree
                ├── tree_1.png
          ├── zebra
                ├── zebra_1.png
    ├── src
          ├── ...         
    ├── inference_single.sh  # Command File for editing images         
    ├── requirements.txt                   
    ├── LICENSE
    └── readme.md

### Execution

```
sh inference_single.sh 
```

You can follow the details of this sh file.
- device_num: your GPU number
- num_ddim_steps: diffusion steps for the inference (default = 50)
- tau: steps of editing images in reverse process (default = 25)
- beta: hyperparameter used in initalization of prompt interpolation (default = 0.3 for word swap / 0.8 for adding phrases)
- gamma: hyperparameter of controlling the corretion term (default = 0.2, gamma * negative_guidance_scale is used in the code.)
- task: translation tasks in the format of "{source phrase}2{target phrase}". For example, if the task is set to 'cat2dog' and source prompt is created as 'a cat is lying on the grass', the target prompt will be set 'a dog is lying on the grass'. Note that the source phrase should be included in the source prompt. You can check the source prompt when you execute this command.


## Citation 
```
@article{lee2024diffusion,
  title={Diffusion-Based Image-to-Image Translation by Noise Correction via Prompt Interpolation},
  author={Lee, Junsung and Kang, Minsoo and Han, Bohyung},
  journal={arXiv preprint arXiv:2409.08077},
  year={2024}
}
```
## License

This project is licensed under the MIT License.
