## Environment
Set Conda Environment:
```bash
conda create -n sbir python==3.12
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install torch torchvision torchaudio
pip install transformers datasets accelerate diffusers 
pip install xformers controlnet_aux
pip install open_clip_torch==2.24.0
pip install einops_exts einops
pip install matplotlib seaborn
```

## Dataset
You can get FS-COCO from [FS-COCO](https://github.com/pinakinathc/fscoco) and SketchyCOCO from [SketchyCOCO](https://github.com/sysu-imsl/SketchyCOCO)
## Evaluate
just change the dataset and path:
```python
python run.py \
--batch_size 4 \
--alpha 0.85 \
--use_optimization \
--retrieval_model_type blip \
--retrieval_model Salesforce/blip-itm-base-coco \
--vllm_model Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5 \
--dataset_file path/to/your/dataset_file \
--dataset_name SketchyCOCO \
--full_images_path ./../data/SketchyCOCO/image_list.csv \
--integrated_captions_save_dir path/to/your/caption_save_dir\
--res_dir path/to/your/result_save_dir
```
